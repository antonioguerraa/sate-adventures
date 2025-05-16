import torch
import sys
import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify, url_for
from enhanced_model import EnhancedLLM

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


# Flask app
app = Flask(__name__)

# Load model and tokenizer - load only once when app starts
print("Loading model...")
device = torch.device('cpu')

# Try to load the enhanced tokenizer if available
tokenizer = CharacterTokenizer()
if os.path.exists('data/enhanced_tokenizer.json'):
    print("Using enhanced tokenizer")
    try:
        import json
        with open('data/enhanced_tokenizer.json', 'r') as f:
            data = json.load(f)
            tokenizer.chars = data['chars']
            tokenizer.stoi = data['stoi']
            tokenizer.itos = {int(k): v for k, v in data['itos'].items()}
    except Exception as e:
        print(f"Error loading enhanced tokenizer: {e}")
        # Fallback to training on Shakespeare text
        with open('data/tinyshakespeare.txt', 'r') as f:
            text = f.read()
        tokenizer.train(text)
else:
    # Fallback to training on Shakespeare text
    with open('data/tinyshakespeare.txt', 'r') as f:
        text = f.read()
    tokenizer.train(text)

# Load checkpoint
# Try to load best available model, with fallbacks
if os.path.exists('models/enhanced_model_best.pt'):
    checkpoint = torch.load('models/enhanced_model_best.pt', map_location=device)
    print("Using enhanced model")
elif os.path.exists('models/simplified_model_best.pt'):
    checkpoint = torch.load('models/simplified_model_best.pt', map_location=device)
    print("Using simplified model")
else:
    checkpoint = torch.load('models/minillm_trained.pt', map_location=device)
    print("Using original model")

# Create model instance
if 'config' in checkpoint:
    # New model format from huggingface_enhanced_train.py
    config = checkpoint['config']
    model = EnhancedLLM(
        vocab_size=config['vocab_size'],
        n_embd=config['n_embd'],
        n_head=config['n_head'],
        n_layer=config['n_layer'],
        block_size=config['block_size'],
        dropout=config.get('dropout', 0.1)  # Default to 0.1 if not specified
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model'])
    print("Loaded enhanced model architecture")
else:
    # Original model format
    model = MiniLLM(
        vocab_size=checkpoint['vocab_size'],
        n_embd=checkpoint['n_embd'],
        n_head=checkpoint['n_head'],
        n_layer=checkpoint['n_layer'],
        block_size=checkpoint['block_size']
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded original model architecture")

model.eval()  # Set to evaluation mode

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.get_json()
    prompt = data.get('prompt', '')
    temperature = float(data.get('temperature', 0.8))
    max_tokens = int(data.get('max_tokens', 100))
    
    try:
        # Safely encode prompt
        encoded_prompt = tokenizer.encode(prompt)
        if not encoded_prompt:  # If encoding returned empty list
            encoded_prompt = tokenizer.encode("The")
            
        # Generate text
        context = torch.tensor([encoded_prompt], dtype=torch.long).to(device)
        generated = model.generate(context, max_tokens, temperature=temperature)[0].tolist()
        generated_text = tokenizer.decode(generated)
        
        return jsonify({
            'generated_text': generated_text,
            'prompt': prompt
        })
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/static/<path:path>')
def static_files(path):
    return app.send_static_file(path)

if __name__ == '__main__':
    print("KING RIZAW's Mini LLM Chat is running!")
    print("Open your browser and go to http://127.0.0.1:5000")
    app.run(debug=False)  # Set debug to False for production