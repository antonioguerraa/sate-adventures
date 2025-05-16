
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
    def __init__(self, vocab_size, n_embd=384, n_head=6, n_layer=6, block_size=256, dropout=0.1):
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
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'enhanced_model_best.pt')
    tokenizer_path = os.path.join(os.path.dirname(__file__), 'data', 'enhanced_tokenizer.json')
    
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
    print(f"\nGenerating from: '{prompt}'\n")
    context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    
    # Generate options
    temperature = 0.8
    max_tokens = 500
    top_k = 50
    
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
