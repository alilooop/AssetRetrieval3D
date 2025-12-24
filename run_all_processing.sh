#!/bin/bash
# Run all processing steps in sequence

set -e  # Exit on error

echo "=== 3D Asset Retrieval System - Full Processing Pipeline ==="
echo ""

# Check if config is valid
echo "Step 0: Validating configuration..."
python config.py
if [ $? -ne 0 ]; then
    echo "❌ Configuration validation failed. Please check your config."
    exit 1
fi
echo "✓ Configuration valid"
echo ""

# Translate captions
echo "Step 1: Translating captions (EN → CN)..."
echo "Note: This may take hours for the full dataset."
read -p "Continue with translation? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python scripts/01_translate_captions.py
    if [ $? -ne 0 ]; then
        echo "❌ Translation failed"
        exit 1
    fi
    echo "✓ Translation complete"
else
    echo "⊘ Skipping translation"
fi
echo ""

# Generate SigLip embeddings
echo "Step 2: Generating SigLip embeddings..."
python scripts/02_embed_siglip.py
if [ $? -ne 0 ]; then
    echo "❌ SigLip embedding generation failed"
    exit 1
fi
echo "✓ SigLip embeddings complete"
echo ""

# Generate Qwen embeddings
echo "Step 3: Generating Qwen embeddings..."
python scripts/03_embed_qwen.py
if [ $? -ne 0 ]; then
    echo "❌ Qwen embedding generation failed"
    exit 1
fi
echo "✓ Qwen embeddings complete"
echo ""

# Populate database
echo "Step 4: Populating databases..."
python scripts/04_populate_database.py
if [ $? -ne 0 ]; then
    echo "❌ Database population failed"
    exit 1
fi
echo "✓ Database population complete"
echo ""

echo "=== ✓ All processing steps completed successfully! ==="
echo ""
echo "Next steps:"
echo "  1. Start backend: ./run_backend.sh (or python backend/app.py)"
echo "  2. Start frontend: ./run_frontend.sh (or python frontend/gradio_app.py)"
echo "  3. Open browser to http://localhost:7860"

