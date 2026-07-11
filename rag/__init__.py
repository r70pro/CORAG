"""
OLMOCR RAG — Retrieval-Augmented Generation for Medicolegal Document Analysis.

This package provides:
- chunker: Medicolegal-aware document chunking with metadata extraction
- embedding: Sentence-transformer embedding + Qdrant vector storage
- retriever: Query embedding + similarity search with metadata filtering
- analyzer: LLM prompt assembly + vLLM OpenAI API integration
- db: PostgreSQL document registry and chunk metadata
- storage: MinIO blob storage for PDFs and markdown
- cache: Redis caching layer for queries and embeddings
"""

__version__ = "0.1.0"
