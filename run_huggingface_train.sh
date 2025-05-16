#!/bin/bash

# Add user's Python bin directory to PATH
export PATH=$PATH:$HOME/Library/Python/3.9/bin

echo "=== Enhanced Mini LLM Training with HuggingFace ====="
echo "This script will train a larger model on Shakespeare, WikiText, and Gutenberg books."
echo "using HuggingFace datasets library for reliable downloading."
echo ""
echo "Key improvements:"
echo "- Increased model size (384 embedding dim, 6 layers, 6 attention heads)"
echo "- Multiple datasets for better language understanding"
echo "- Advanced architecture with pre-norm transformers, dropout, and GELU activations"
echo "- Better training with learning rate scheduling and weight decay"
echo "- Top-k sampling for improved text generation"
echo ""
echo "Total parameters: ~3.5M (compared to 160K in the original model)"
echo ""
echo "Press Enter to start training, or Ctrl+C to cancel..."
read

# Create necessary directories
mkdir -p data models

# Run the enhanced training script
python3 huggingface_enhanced_train.py

# Training complete message
echo ""
echo "Training complete! To generate text with your enhanced model:"
echo "python3 enhanced_generate.py \"Your prompt here\""