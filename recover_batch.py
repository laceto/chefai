"""
recover_batch.py
One-off recovery script: resume a completed OpenAI batch job without resubmitting.

Use this when embed_recipes.py crashed AFTER the batch completed but BEFORE
the vectorstore and registry were saved.  OpenAI retains batch results for 24 h.

Usage:
    python recover_batch.py --job-id batch_69c5164811688190a9a96fc0274d364a

    # dry-run: parse and align only, skip writing the store and registry
    python recover_batch.py --job-id batch_xxx --dry-run

Environment:
    OPENAI_API_KEY  — required (loaded from .env)
    LOG_LEVEL       — optional, default INFO
"""

import argparse
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
    init_vectorstore,
    load_all_recipes,
    load_registry,
    save_registry,
    update_vectorstore,
)
from kitai.batch import download_batch_results, parse_embedding_results

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger(__name__)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job-id", required=True,
                        help="Completed OpenAI batch job ID to recover from.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse and align only — do not write vectorstore or registry.")
    args = parser.parse_args()

    load_dotenv()
    client           = OpenAI()
    embeddings_model = OpenAIEmbeddings(
        model=DEFAULT_EMBED_MODEL, dimensions=DEFAULT_EMBED_DIMS
    )

    # 1. Reconstruct the same docs that were submitted to the batch
    registry = load_registry()
    all_df   = load_all_recipes()
    known    = set(registry["recipe_id"]) if not registry.empty else set()
    new_df   = all_df[~all_df["recipe_id"].isin(known)].copy()

    if new_df.empty:
        log.info("Registry is already up to date — nothing to recover.")
        sys.exit(0)

    new_df = assign_ids(new_df, registry)
    docs   = build_documents(new_df)
    log.info("Reconstructed %d documents for alignment.", len(docs))

    # 2. Download results from the completed batch job
    log.info("Downloading results for job %s ...", args.job_id)
    results = download_batch_results(client, args.job_id)
    pairs   = parse_embedding_results(results)
    log.info("Parsed %d embeddings from job %s.", len(pairs), args.job_id)

    # 3. Align embeddings to doc order; drop any API-level failures
    aligned_pairs, aligned_docs = align_pairs_to_docs(pairs, docs)

    if not aligned_docs:
        log.error("All embeddings missing — nothing to write. Exiting.")
        sys.exit(1)

    if args.dry_run:
        log.info("[dry-run] Would write %d vectors. Exiting without changes.",
                 len(aligned_docs))
        sys.exit(0)

    # 4. Init or incrementally update the vectorstore
    if VECTORSTORE_DIR.exists():
        store = update_vectorstore(aligned_pairs, aligned_docs, embeddings_model)
    else:
        store = init_vectorstore(aligned_docs, aligned_pairs, embeddings_model)

    # 5. Persist store BEFORE writing registry (same ordering invariant as embed_recipes.py)
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(VECTORSTORE_DIR))
    log.info("Vectorstore saved to %s.", VECTORSTORE_DIR)

    # 6. Append recovered rows to the registry
    new_rows = pd.DataFrame({
        "id":          [doc.metadata["id"]          for doc in aligned_docs],
        "recipe_id":   [doc.metadata["recipe_id"]   for doc in aligned_docs],
        "title":       [doc.metadata["title"]       for doc in aligned_docs],
        "source_file": [doc.metadata["source_file"] for doc in aligned_docs],
    })
    updated_registry = pd.concat([registry, new_rows], ignore_index=True)
    save_registry(updated_registry)

    log.info(
        "[done] Recovered %d recipes. Total in registry: %d.",
        len(aligned_docs),
        len(updated_registry),
    )


if __name__ == "__main__":
    main()
