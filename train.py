import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import time
import os
from pathlib import Path
import urllib.request

# Simplified tokenizer implementation
class CharacterTokenizer:
    def __init__(self):
        self.chars = []
        self.stoi = {}  # string to index
        self.itos = {}  # index to string
        
    def train(self, text):
        # Get unique characters from the text
        chars = sorted(list(set(text)))
        self.chars = chars
        
        # Create mappings
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        return self
    
    def encode(self, text):
        # Convert string to list of integers
        return [self.stoi[c] for c in text]
    
    def decode(self, ids):
        # Convert list of integers to string
        return ''.join([self.itos[i] for i in ids])
    
    def vocab_size(self):
        return len(self.chars)

# Simplified model implementation
class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        
        # compute attention scores
        wei = q @ k.transpose(-2, -1) * (C ** -0.5) # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        
        # weighted aggregation of values
        v = self.value(x) # (B,T,head_size)
        out = wei @ v # (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )
        
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MiniLLM(nn.Module):
    def __init__(self, vocab_size, n_embd=64, n_head=4, n_layer=3, block_size=256):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Token and position embeddings
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        
        # Apply transformer blocks
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        
        # Apply language modeling head
        logits = self.lm_head(x)  # (B,T,vocab_size)
        
        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop context if it exceeds block_size
            idx_cond = idx[:, -self.block_size:]
            
            # Get the predictions
            logits, _ = self(idx_cond)
            
            # Focus only on the last time step
            logits = logits[:, -1, :] / temperature  # (B, C)
                
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
            
        return idx

# Download tiny Shakespeare dataset
def download_shakespeare():
    """Download tiny Shakespeare dataset if not present"""
    data_path = Path("data")
    data_path.mkdir(exist_ok=True)
    
    shakespeare_path = data_path / "tinyshakespeare.txt"
    
    if not shakespeare_path.exists():
        print("Downloading tiny Shakespeare dataset...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, shakespeare_path)
        print(f"Downloaded to {shakespeare_path}")
    
    return shakespeare_path

def train():
    # Training hyperparameters - minimized for faster training
    batch_size = 32
    block_size = 64  # Context length
    learning_rate = 1e-3
    max_iters = 1000  # Reduced for quick training
    n_embd = 64
    n_head = 4
    n_layer = 3
    
    # Setup device - force CPU for compatibility
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Data directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Get Shakespeare data
    shakespeare_path = download_shakespeare()
    with open(shakespeare_path, 'r') as f:
        text = f.read()
    print(f"Text length: {len(text):,} characters")
    
    # Initialize and train the tokenizer
    tokenizer = CharacterTokenizer().train(text)
    vocab_size = tokenizer.vocab_size()
    print(f"Vocabulary size: {vocab_size} unique characters")
    
    # Encode the entire text
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    
    # Split data
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    # Data loading function
    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x.to(device), y.to(device)
    
    # Initialize the model
    model = MiniLLM(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size
    ).to(device)
    
    # Number of parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {n_params:,}")
    
    # Initialize the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    print(f"Starting training for {max_iters} iterations...")
    start_time = time.time()
    
    for iter in range(max_iters):
        if iter % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Iteration {iter}/{max_iters}, {elapsed:.2f}s elapsed")
        
        # Sample a batch
        xb, yb = get_batch('train')
        
        # Evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # Generate text every 200 iterations
        if iter % 200 == 0 or iter == max_iters - 1:
            print(f"Iteration {iter}, Loss: {loss.item():.4f}")
            # Generate some text
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            context[0, 0] = tokenizer.encode('.')[0]  # Start with a period
            generated_text = model.generate(context, max_new_tokens=100, temperature=0.8)[0].tolist()
            decoded_text = tokenizer.decode(generated_text)
            print(f"\nGenerated sample:\n{decoded_text}\n")
    
    # Save model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "minillm_trained.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'n_embd': n_embd,
        'n_head': n_head,
        'n_layer': n_layer,
        'block_size': block_size,
    }, model_path)
    
    print(f"Training complete! Model saved to {model_path}")
    
    # Final generation demo
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    context[0, 0] = tokenizer.encode('The')[0]  # Start with "The"
    print("\nGenerating final sample...")
    generated_text = model.generate(context, max_new_tokens=500, temperature=0.8)[0].tolist()
    decoded_text = tokenizer.decode(generated_text)
    print(f"\nFinal generated text:\n{decoded_text}\n")
    
    # Create a simple script to generate text
    Path("generate_minimal.py").write_text("""
import torch
import sys
from pathlib import Path

# Character tokenizer
class CharacterTokenizer:
    def __init__(self):
        self.chars = []
        self.stoi = {}
        self.itos = {}
        
    def train(self, text):
        chars = sorted(list(set(text)))
        self.chars = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        return self
    
    def encode(self, text):
        return [self.stoi[c] for c in text]
    
    def decode(self, ids):
        return ''.join([self.itos[i] for i in ids])
    
    def vocab_size(self):
        return len(self.chars)

# Load model
device = torch.device('cpu')
checkpoint = torch.load('models/minillm_trained.pt', map_location=device)

# Get Shakespeare text for tokenizer
with open('data/tinyshakespeare.txt', 'r') as f:
    text = f.read()

# Initialize tokenizer
tokenizer = CharacterTokenizer().train(text)

# Get model architecture
from train_minimal import MiniLLM

# Create model instance
model = MiniLLM(
    vocab_size=checkpoint['vocab_size'],
    n_embd=checkpoint['n_embd'],
    n_head=checkpoint['n_head'],
    n_layer=checkpoint['n_layer'],
    block_size=checkpoint['block_size']
).to(device)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Get prompt from user
prompt = sys.argv[1] if len(sys.argv) > 1 else "Once upon a time"
print(f"Using prompt: '{prompt}'")

# Generate text
context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
generated = model.generate(context, max_new_tokens=500, temperature=0.8)[0].tolist()
generated_text = tokenizer.decode(generated)
print("\\nGenerated text:\\n" + "-" * 50)
print(generated_text)
print("-" * 50)
""")
    
    print("\nCreated generate_minimal.py for text generation")
    print("Use it with: python generate_minimal.py \"Your prompt here\"")
    
    return model, tokenizer

if __name__ == "__main__":
    train()