#!/bin/bash

# Add user's Python bin directory to PATH
export PATH=$PATH:$HOME/Library/Python/3.9/bin

echo "=== Quick Enhanced Mini LLM Training ==="
echo "This script will train a moderately-sized LLM in about 20-30 minutes."
echo ""
echo "Model specifications:"
echo "- 128 embedding dimensions"
echo "- 4 transformer layers"
echo "- 4 attention heads"
echo "- Total parameters: ~0.5M"
echo ""
echo "This training uses:"
echo "- Shakespeare's works"
echo "- Small sample of classic literature"
echo "- Top-k sampling for better generation"
echo ""
echo "Press Enter to start training, or Ctrl+C to cancel..."
read

# Create necessary directories
mkdir -p data models

# Run the simplified training script
python3 simplified_enhanced_train.py

# Training complete message
echo ""
echo "Training complete! To generate text with your model:"
echo "python3 simplified_generate.py \"Your prompt here\""