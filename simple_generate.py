import torch
import sys
from pathlib import Path

# Model architecture
class Head(torch.nn.Module):
    def __init__(self, head_size, n_embd, block_size):
        super().__init__()
        self.key = torch.nn.Linear(n_embd, head_size, bias=False)
        self.query = torch.nn.Linear(n_embd, head_size, bias=False)
        self.value = torch.nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = torch.nn.functional.softmax(wei, dim=-1)
        
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size):
        super().__init__()
        self.heads = torch.nn.ModuleList([Head(head_size, n_embd, block_size) for _ in range(num_heads)])
        self.proj = torch.nn.Linear(head_size * num_heads, n_embd)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(torch.nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_embd, 4 * n_embd),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * n_embd, n_embd),
        )
        
    def forward(self, x):
        return self.net(x)

class Block(torch.nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = torch.nn.LayerNorm(n_embd)
        self.ln2 = torch.nn.LayerNorm(n_embd)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MiniLLM(torch.nn.Module):
    def __init__(self, vocab_size, n_embd=64, n_head=4, n_layer=3, block_size=256):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = torch.nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = torch.nn.Embedding(block_size, n_embd)
        self.blocks = torch.nn.Sequential(*[Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = torch.nn.LayerNorm(n_embd)
        self.lm_head = torch.nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        
        x = self.blocks(x)
        x = self.ln_f(x)
        
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            
            logits, _ = self(idx_cond)
            
            logits = logits[:, -1, :] / temperature
                
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            idx_next = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx

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
        # Safely encode text, skipping unknown characters
        return [self.stoi[c] for c in text if c in self.stoi]
    
    def decode(self, ids):
        # Safely decode indices
        return ''.join([self.itos.get(i, '') for i in ids])

def main():
    print("Loading model...")
    # Determine device
    device = torch.device('cpu')
    
    # Get Shakespeare text for tokenizer
    with open('data/tinyshakespeare.txt', 'r') as f:
        text = f.read()
    
    # Initialize tokenizer
    tokenizer = CharacterTokenizer().train(text)
    
    # Load checkpoint
    checkpoint = torch.load('models/minillm_trained.pt', map_location=device)
    
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
    model.eval()  # Set to evaluation mode
    
    # Get prompt
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Once upon a time"
    print(f"Using prompt: '{prompt}'")
    
    try:
        # Safely encode prompt
        encoded_prompt = tokenizer.encode(prompt)
        if not encoded_prompt:  # If encoding returned empty list
            print("Warning: Prompt contains characters not seen during training. Using default.")
            encoded_prompt = tokenizer.encode("The")
            
        # Generate text
        context = torch.tensor([encoded_prompt], dtype=torch.long).to(device)
        generated = model.generate(context, max_new_tokens=500, temperature=0.8)[0].tolist()
        generated_text = tokenizer.decode(generated)
        
        print("\nGenerated text:\n" + "-" * 50)
        print(generated_text)
        print("-" * 50)
    except Exception as e:
        print(f"Error during generation: {str(e)}")

if __name__ == "__main__":
    main()