"""
embed_recipes.py
Entry-point script: builds and incrementally updates a FAISS vectorstore
from the recipe corpus (data/recipes.json) using the OpenAI Batch API via kitai.

All logic lives in chefai.feed_embedder — import from there for programmatic use.

Behaviour:
- Cold start (no existing vectorstore): embeds ALL recipes and creates
  the store from scratch.
- Incremental (store already exists): embeds only recipes whose recipe_id is not
  yet in the registry, then appends them to the existing store via
  FAISS.add_embeddings (no rebuild required).
- No-op: exits 0 cleanly if no new recipes are found.

Run:
    python embed_recipes.py

Environment:
    OPENAI_API_KEY  — required (loaded from .env)
    LOG_LEVEL       — optional, default INFO (set to DEBUG for batch poll detail)
"""

import logging
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

from constants import VECTORSTORE_DIR
from chefai.feed_embedder import (
    DEFAULT_EMBED_DIMS,
    DEFAULT_EMBED_MODEL,
    align_pairs_to_docs,
    assign_ids,
    build_documents,
    find_new_recipes,
    init_vectorstore,
    load_all_recipes,
    load_registry,
    run_embedding_batch,
    save_registry,
    update_vectorstore,
)

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger(__name__)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv()
    client           = OpenAI()
    embeddings_model = OpenAIEmbeddings(
        model=DEFAULT_EMBED_MODEL, dimensions=DEFAULT_EMBED_DIMS
    )

    # 1. Load existing state
    registry = load_registry()

    # 2. Discover new recipes (exits 0 if none)
    all_df = load_all_recipes()
    new_df = find_new_recipes(all_df, registry)
    new_df = assign_ids(new_df, registry)

    # 3. Build LangChain Documents
    docs = build_documents(new_df)

    # 4. Embed via OpenAI Batch API
    pairs = run_embedding_batch(docs, client)

    # 5. Align embeddings to doc order; drop any API-level failures
    aligned_pairs, aligned_docs = align_pairs_to_docs(pairs, docs)

    if not aligned_docs:
        log.error("All embeddings failed — nothing to add to the store. Exiting.")
        sys.exit(1)

    # 6. Init or incrementally update the vectorstore
    if VECTORSTORE_DIR.exists():
        store = update_vectorstore(aligned_pairs, aligned_docs, embeddings_model)
    else:
        store = init_vectorstore(aligned_docs, aligned_pairs, embeddings_model)

    # 7. Persist the store BEFORE writing the registry.
    #    If save_local fails, the registry stays at its previous consistent state
    #    and the next run will re-embed the same articles.
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(VECTORSTORE_DIR))
    log.info("Vectorstore saved to %s.", VECTORSTORE_DIR)

    # 8. Append new rows to the registry (atomic write)
    new_rows = pd.DataFrame({
        "id":          [doc.metadata["id"]          for doc in aligned_docs],
        "recipe_id":   [doc.metadata["recipe_id"]   for doc in aligned_docs],
        "title":       [doc.metadata["title"]       for doc in aligned_docs],
        "source_file": [doc.metadata["source_file"] for doc in aligned_docs],
    })
    updated_registry = pd.concat([registry, new_rows], ignore_index=True)
    save_registry(updated_registry)

    log.info(
        "[done] Embedded %d new recipes. Total in registry: %d.",
        len(aligned_docs),
        len(updated_registry),
    )


if __name__ == "__main__":
    main()
