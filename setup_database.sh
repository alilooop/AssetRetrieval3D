#!/bin/bash
# Setup PostgreSQL databases and enable pgvector extension

echo "Setting up PostgreSQL databases..."

# Note: Make sure PostgreSQL is running and you have proper credentials

echo "Creating databases..."
psql -U ${DB_USER:-postgres} -h ${DB_HOST:-localhost} -c "CREATE DATABASE siglip_embeddings;" 2>/dev/null || echo "Database siglip_embeddings may already exist"
psql -U ${DB_USER:-postgres} -h ${DB_HOST:-localhost} -c "CREATE DATABASE qwen_embeddings;" 2>/dev/null || echo "Database qwen_embeddings may already exist"

echo "Enabling pgvector extension..."
psql -U ${DB_USER:-postgres} -h ${DB_HOST:-localhost} -d siglip_embeddings -c "CREATE EXTENSION IF NOT EXISTS vector;"
psql -U ${DB_USER:-postgres} -h ${DB_HOST:-localhost} -d qwen_embeddings -c "CREATE EXTENSION IF NOT EXISTS vector;"

echo "✓ Database setup complete!"

