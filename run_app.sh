#!/bin/bash

# Add user's Python bin directory to PATH
export PATH=$PATH:$HOME/Library/Python/3.9/bin

echo "Creating necessary directories if they don't exist..."
mkdir -p static/images

echo "Checking Python packages..."
pip3 install torch flask

echo "Starting Mini LLM Chat Interface..."
echo "Open your browser and go to http://127.0.0.1:5000"
echo "Press Ctrl+C to stop the server"

# Run the Flask app
python3 app.py