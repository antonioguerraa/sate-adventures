# Enhanced Mini LLM

An improved language model implementation with multiple data sources, optimized for Apple Silicon.

## Features

- Transformer-based architecture with self-attention mechanisms
- Training on multiple datasets (Shakespeare, WikiText, Gutenberg books)
- Enhanced model parameters (~3.5M parameters vs 160K in original)
- Text generation with temperature and top-k sampling
- Optimized for Apple Silicon M-series chips

## Models

This project includes two model options:

### Original Mini LLM (160K parameters)
- Embedding dimension: 64
- Number of attention heads: 4
- Number of transformer layers: 3
- Context window: 64 tokens
- Trained on Shakespeare text only

### Enhanced Mini LLM (3.5M parameters)
- Embedding dimension: 384
- Number of attention heads: 6
- Number of transformer layers: 6
- Context window: 256 tokens
- Trained on Shakespeare, WikiText, and Gutenberg books
- Learning rate scheduling and weight decay for better training
- Advanced architecture with pre-norm transformers and dropout

## Setup

```bash
# Install dependencies
pip3 install torch numpy matplotlib tqdm
```

## Usage

### Generate text with basic model:
```bash
python3 simple_generate.py "Your prompt here"
```

### Generate text with enhanced model (after training):
```bash
python3 enhanced_generate.py "Your prompt here"
```

### Train the enhanced model:
```bash
./run_enhanced_training.sh
```

### Run web interface (after training enhanced model):
```bash
./run_app.sh
```

## Web Interface

The project includes a modern Apple-style chat interface that allows you to interact with your trained model through a web browser.

Features:
- Clean, iOS-inspired design with rounded corners
- Light and dark mode support
- Adjustable generation parameters
- Real-time text generation

## Examples

Here's a comparison of outputs from both models:

**Original Model (Shakespeare only):**
```
Prompt: "To be or not to be"
Output: "To be or not to be pof I threvith,
I to and mon with ben parkent thour year that were mecichar.
Wat opeard, to that it stat of theard come ond comor of hild..."
```

**Enhanced Model (Multiple datasets):**
*[Examples will appear after training]*

## Architecture

The enhanced model uses pre-normalization transformer blocks with multi-head attention and dropout regularization. It implements modern techniques like:

- GELU activation functions
- Weight decay for regularization
- Learning rate warmup and decay
- Top-k sampling for generation
- Extended context window

## Training Data

The model is trained on a combination of:
1. Shakespeare's complete works
2. WikiText-2 dataset (high-quality Wikipedia articles)
3. Classic literature from Project Gutenberg (Pride and Prejudice, Moby Dick, etc.)