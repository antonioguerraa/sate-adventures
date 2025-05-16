#!/bin/bash

echo "Setting up Mini LLM environment..."

# Install requirements
pip3 install torch numpy

echo "Installation complete!"
echo "To generate text, run: python3 simple_generate.py \"Your prompt here\""
echo "Enjoy your tiny Shakespeare-inspired language model!"