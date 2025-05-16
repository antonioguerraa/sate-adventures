import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import time
import os
from pathlib import Path
import urllib.request
import random
import math
from tqdm import tqdm

# Enhanced tokenizer implementation
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
        # Convert string to list of integers, handling unknown chars
        return [self.stoi[c] if c in self.stoi else random.randint(0, len(self.chars)-1) for c in text]
    
    def decode(self, ids):
        # Convert list of integers to string
        return ''.join([self.itos.get(i, '') for i in ids])
    
    def vocab_size(self):
        return len(self.chars)
        
    def save(self, path):
        import json
        with open(path, 'w') as f:
            json.dump({
                'chars': self.chars,
                'stoi': self.stoi,
                'itos': {str(k): v for k, v in self.itos.items()}  # Convert int keys to strings for JSON
            }, f)
            
    def load(self, path):
        import json
        with open(path, 'r') as f:
            data = json.load(f)
            self.chars = data['chars']
            self.stoi = data['stoi']
            self.itos = {int(k): v for k, v in data['itos'].items()}  # Convert string keys back to ints
        return self

# Enhanced model with more advanced architecture
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0, "embedding dimension must be divisible by number of heads"
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = n_embd // n_head
        
        # Key, Query, Value projections
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        
        # Regularization
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        
        # Output projection
        self.proj = nn.Linear(n_embd, n_embd)
        
        # Causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self, x):
        B, T, C = x.shape  # batch size, sequence length, embedding dimension
        
        # Calculate query, key, values for all heads in batch
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        
        # Attention: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # Apply causal mask
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        
        # Apply attention to values
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        
        # Output projection
        y = self.resid_drop(self.proj(y))
        return y

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),  # Using GELU instead of ReLU for better performance
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout)
        
    def forward(self, x):
        # Pre-norm architecture (better stability)
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class EnhancedLLM(nn.Module):
    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=4, block_size=128, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        
        # Embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        
        # Final layer norm and head
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Report number of parameters
        print(f"Number of parameters: {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M")
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Embed tokens and positions
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, C)
        x = self.dropout(tok_emb + pos_emb)  # (B, T, C)
        
        # Apply transformer blocks
        x = self.blocks(x)
        x = self.ln_f(x)
        
        # Get logits
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop context if it exceeds block_size
            idx_cond = idx[:, -self.block_size:]
            
            # Get predictions
            logits, _ = self(idx_cond)
            
            # Focus only on the last time step
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx

# Data download functions
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

def download_gutenberg():
    """Download a selection of Gutenberg books if not present"""
    data_path = Path("data")
    data_path.mkdir(exist_ok=True)
    
    gutenberg_path = data_path / "gutenberg.txt"
    
    if not gutenberg_path.exists():
        # Create a dummy sample for now since we're in a hurry
        print("Creating sample Gutenberg text...")
        sample_text = """
        It was the best of times, it was the worst of times,
        it was the age of wisdom, it was the age of foolishness,
        it was the epoch of belief, it was the epoch of incredulity,
        it was the season of Light, it was the season of Darkness,
        it was the spring of hope, it was the winter of despair.
        
        Call me Ishmael. Some years ago—never mind how long precisely—having
        little or no money in my purse, and nothing particular to interest me on shore,
        I thought I would sail about a little and see the watery part of the world.
        
        In my younger and more vulnerable years my father gave me some advice that I've
        been turning over in my mind ever since. "Whenever you feel like criticizing any one,"
        he told me, "just remember that all the people in this world haven't had the advantages
        that you've had."
        
        Happy families are all alike; every unhappy family is unhappy in its own way.
        
        It is a truth universally acknowledged, that a single man in possession of a good fortune,
        must be in want of a wife.
        
        For a long time, I went to bed early. Sometimes, my candle scarcely out, my eyes would
        close so quickly that I did not have time to say to myself: "I'm falling asleep."
        """
        
        with open(gutenberg_path, 'w') as f:
            f.write(sample_text)
        
        print(f"Created sample text at {gutenberg_path}")
    
    return gutenberg_path

def train():
    # Training hyperparameters - simplified for faster training
    batch_size = 32
    block_size = 128  # Context length
    learning_rate = 5e-4
    max_iters = 500  # Reduced for quick training
    eval_interval = 100
    eval_batches = 10
    
    # Model parameters - reduced for faster training
    n_embd = 128
    n_head = 4
    n_layer = 4
    dropout = 0.1
    
    # Setup device - use CUDA if available, otherwise fallback to CPU or MPS (Apple Silicon)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA...")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Apple Silicon)...")
    else:
        device = torch.device('cpu')
        print("Using CPU...")
    
    # Data directory setup
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Download datasets
    shakespeare_path = download_shakespeare()
    print(f"Shakespeare data: {shakespeare_path}")
    
    gutenberg_path = download_gutenberg()
    print(f"Gutenberg sample data: {gutenberg_path}")
    
    # Combine texts
    all_text = ""
    
    # Add Shakespeare text
    with open(shakespeare_path, 'r', encoding='utf-8', errors='replace') as f:
        all_text += f.read()
    print(f"Shakespeare text length: {len(all_text):,} characters")
    
    # Add Gutenberg text
    with open(gutenberg_path, 'r', encoding='utf-8', errors='replace') as f:
        all_text += f.read()
    print(f"Final text length (with Gutenberg): {len(all_text):,} characters")
    
    # Initialize and train the tokenizer
    tokenizer = CharacterTokenizer().train(all_text)
    vocab_size = tokenizer.vocab_size()
    print(f"Vocabulary size: {vocab_size} unique characters")
    
    # Save tokenizer
    tokenizer.save(data_dir / 'simplified_tokenizer.json')
    
    # Encode the entire text
    data = torch.tensor(tokenizer.encode(all_text), dtype=torch.long)
    
    # Split data - 90% train, 10% validation
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    print(f"Train data: {len(train_data):,} tokens")
    print(f"Validation data: {len(val_data):,} tokens")
    
    # Simple data loading function
    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x.to(device), y.to(device)
    
    # Initialize the model
    model = EnhancedLLM(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size,
        dropout=dropout
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    
    print(f"Starting training for {max_iters} iterations...")
    
    start_time = time.time()
    progress_bar = tqdm(range(max_iters), desc="Training")
    
    try:
        for iter in progress_bar:
            # Sample a batch of data
            xb, yb = get_batch('train')
            
            # Evaluate the loss
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # Evaluation
            if iter % eval_interval == 0 or iter == max_iters - 1:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for _ in range(eval_batches):
                        eval_x, eval_y = get_batch('val')
                        _, loss = model(eval_x, eval_y)
                        val_loss += loss.item()
                
                val_loss /= eval_batches
                
                print(f"\nIteration {iter}: train loss {loss.item():.4f}, val loss {val_loss:.4f}")
                
                # Generate sample text
                context = torch.zeros((1, 1), dtype=torch.long, device=device)
                context[0, 0] = tokenizer.encode('.')[0]  # Start with a period
                sample = model.generate(context, max_new_tokens=100, temperature=0.8, top_k=40)[0].tolist()
                print(f"\nGenerated sample:\n{tokenizer.decode(sample)}\n")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"Saving best model with val loss: {best_val_loss:.4f}")
                    checkpoint = {
                        'model': model.state_dict(),
                        'config': {
                            'vocab_size': vocab_size,
                            'n_embd': n_embd,
                            'n_head': n_head,
                            'n_layer': n_layer,
                            'block_size': block_size,
                            'dropout': dropout
                        }
                    }
                    torch.save(checkpoint, models_dir / 'simplified_model_best.pt')
                
                # Resume training
                model.train()
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Save final model
    checkpoint = {
        'model': model.state_dict(),
        'config': {
            'vocab_size': vocab_size,
            'n_embd': n_embd,
            'n_head': n_head,
            'n_layer': n_layer,
            'block_size': block_size,
            'dropout': dropout
        }
    }
    torch.save(checkpoint, models_dir / 'simplified_model_final.pt')
    print(f"Final model saved.")
    
    # Create standalone inference script
    inference_script = """
import torch
import json
import sys
import os
from pathlib import Path

# Model architecture
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0, "embedding dimension must be divisible by number of heads"
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = n_embd // n_head
        
        # Key, Query, Value projections
        self.key = torch.nn.Linear(n_embd, n_embd)
        self.query = torch.nn.Linear(n_embd, n_embd)
        self.value = torch.nn.Linear(n_embd, n_embd)
        
        # Regularization
        self.attn_drop = torch.nn.Dropout(dropout)
        self.resid_drop = torch.nn.Dropout(dropout)
        
        # Output projection
        self.proj = torch.nn.Linear(n_embd, n_embd)
        
        # Causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self, x):
        B, T, C = x.shape  # batch size, sequence length, embedding dimension
        
        # Calculate query, key, values for all heads in batch
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        
        # Attention: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
        # Apply causal mask
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        att = torch.nn.functional.softmax(att, dim=-1)
        att = self.attn_drop(att)
        
        # Apply attention to values
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        
        # Output projection
        y = self.resid_drop(self.proj(y))
        return y

class FeedForward(torch.nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_embd, 4 * n_embd),
            torch.nn.GELU(),  # Using GELU instead of ReLU for better performance
            torch.nn.Linear(4 * n_embd, n_embd),
            torch.nn.Dropout(dropout),
        )
        
    def forward(self, x):
        return self.net(x)

class Block(torch.nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = torch.nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout)
        
    def forward(self, x):
        # Pre-norm architecture (better stability)
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class EnhancedLLM(torch.nn.Module):
    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=4, block_size=128, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        
        # Embeddings
        self.token_embedding_table = torch.nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = torch.nn.Embedding(block_size, n_embd)
        self.dropout = torch.nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = torch.nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        
        # Final layer norm and head
        self.ln_f = torch.nn.LayerNorm(n_embd)
        self.lm_head = torch.nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Embed tokens and positions
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, C)
        x = self.dropout(tok_emb + pos_emb)  # (B, T, C)
        
        # Apply transformer blocks
        x = self.blocks(x)
        x = self.ln_f(x)
        
        # Get logits
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = torch.nn.functional.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop context if it exceeds block_size
            idx_cond = idx[:, -self.block_size:]
            
            # Get predictions
            logits, _ = self(idx_cond)
            
            # Focus only on the last time step
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx

# Character tokenizer for encoder/decoder
class CharacterTokenizer:
    def __init__(self):
        self.chars = []
        self.stoi = {}
        self.itos = {}
    
    def encode(self, text):
        # Safely encode text, handling unknown chars
        return [self.stoi[c] if c in self.stoi else 0 for c in text]
    
    def decode(self, ids):
        # Convert list of integers to string
        return ''.join([self.itos.get(str(i), '') for i in ids])
    
    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
            self.chars = data['chars']
            self.stoi = data['stoi']
            self.itos = data['itos']
        return self

def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA...")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Apple Silicon)...")
    else:
        device = torch.device('cpu')
        print("Using CPU...")
    
    print("Loading model and tokenizer...")
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'simplified_model_best.pt')
    tokenizer_path = os.path.join(os.path.dirname(__file__), 'data', 'simplified_tokenizer.json')
    
    # Load tokenizer
    tokenizer = CharacterTokenizer().load(tokenizer_path)
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    # Create model instance
    model = EnhancedLLM(
        vocab_size=config['vocab_size'],
        n_embd=config['n_embd'],
        n_head=config['n_head'],
        n_layer=config['n_layer'],
        block_size=config['block_size'],
        dropout=config['dropout']
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Get prompt from command line
    if len(sys.argv) > 1:
        prompt = ' '.join(sys.argv[1:])
    else:
        prompt = input("Enter a prompt: ")
    
    # Generate text
    print(f"\\nGenerating from: '{prompt}'\\n")
    context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    
    # Generate options
    temperature = 0.8
    max_tokens = 300
    top_k = 40
    
    # Generate text
    with torch.no_grad():
        generated = model.generate(
            context, 
            max_new_tokens=max_tokens, 
            temperature=temperature,
            top_k=top_k
        )[0].tolist()
    
    # Print generated text
    print("=" * 40)
    print(tokenizer.decode(generated))
    print("=" * 40)

if __name__ == "__main__":
    main()
"""
    
    with open("simplified_generate.py", "w") as f:
        f.write(inference_script)
    
    print("Created simplified_generate.py script for inference.")
    print("Training complete!")
    
    # Generate final samples
    model.eval()
    print("\nGenerating final samples...")
    
    prompts = [
        "Once upon a time",
        "The meaning of life",
        "To be or not to be"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: \"{prompt}\"")
        context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
        sample = model.generate(context, max_new_tokens=200, temperature=0.8, top_k=40)[0].tolist()
        print(f"Generated text:\n{tokenizer.decode(sample)}")
    
    return model, tokenizer

if __name__ == "__main__":
    train()