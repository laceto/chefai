"""
constants.py
Single source of truth for shared taxonomy, scores, and file paths.

Import pattern:

"""

from pathlib import Path
from typing import Literal


# ── File paths ──────────────────────────────────────────────────────────────────

RAW_FEED_DIR       = Path("output")
PENDING_BATCH_FILE  = Path("data/embeddings") / "pending_sector_batch.txt"
BATCH_FILE          = Path("data/embeddings") / "batch_tasks_sector.jsonl"

# ── Recipe corpus ────────────────────────────────────────────────────────────
# Built by scripts/build_corpus.py from data/processed/*.md files.

RECIPES_CORPUS_FILE = Path("data") / "recipes.json"

# ── Recipe vectorstore ────────────────────────────────────────────────────────
# FAISS index and registry produced by embed_recipes.py.
# Change VECTORSTORE_DIR here to move the store; embed_recipes.py picks it up.

VECTORSTORE_DIR     = Path("data/embeddings") / "vectorstore" / "recipes_faiss"
FEEDS_REGISTRY_FILE = Path("data/embeddings") / "vectorstore" / "recipes_registry.tsv"
