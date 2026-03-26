"""
chefai.feed_embedder
Reusable helpers for building and incrementally updating a FAISS vectorstore
from RSS feed articles using the OpenAI Batch API via kitai.

Public API
----------
Constants:
    REGISTRY_COLUMNS      — TSV column order for the article registry
    DEFAULT_EMBED_MODEL   — "text-embedding-3-small"
    DEFAULT_EMBED_DIMS    — 1536
    DEFAULT_POLL_INTERVAL — 30 (seconds between batch-status polls)

Registry:
    load_registry()               -> pd.DataFrame
    save_registry(registry)       -> None

Feed loading:
    load_all_feed_files()                    -> pd.DataFrame
    find_new_articles(all_df, registry)      -> pd.DataFrame
    assign_ids(new_df, registry)             -> pd.DataFrame

Document building:
    build_documents(new_df)                  -> list[Document]

Embedding batch:
    run_embedding_batch(docs, client, *, embed_model, poll_interval)
                                             -> list[tuple[str, list[float]]]
    align_pairs_to_docs(pairs, docs)         -> (aligned_pairs, aligned_docs)

Vectorstore:
    init_vectorstore(docs, text_emb_pairs, embeddings_model)   -> FAISS
    update_vectorstore(text_emb_pairs, aligned_docs,
                       embeddings_model)                        -> FAISS

Invariants
----------
- metadata["id"] is a monotonic integer, never reused across runs.
- feeds_registry.tsv is saved AFTER store.save_local() — a write failure
  leaves the registry consistent with the previous successful state.
- The registry's guid column is the single source of truth for what is in
  the FAISS store. FAISS contents ≡ guids in feeds_registry.tsv.
- Deduplication key: guid (stable RSS-feed identifier per article).
- custom_id in embedding tasks == raw doc.metadata["id"] value (no prefix).

Failure modes
-------------
- Batch API error/expiry  → RuntimeError raised; registry untouched.
- Partial batch failure   → affected docs dropped silently; retried next run.
- FAISS save fails        → registry not written; consistent state preserved.
- All embeddings fail     → caller receives ([], []) and decides how to exit.

Note: load_all_feed_files and find_new_articles call sys.exit(0) for no-op
conditions (no files found / nothing new).  This is intentional for CLI use;
callers that need programmatic control should guard those cases themselves.
"""

import logging
import sys

import numpy as np
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

from constants import FEEDS_REGISTRY_FILE, RAW_FEED_DIR, VECTORSTORE_DIR
from kitai.batch import (
    build_embedding_tasks,
    download_batch_results,
    parse_embedding_results,
    poll_until_complete,
    submit_batch_job,
)
from kitai.index import create_vectorstore

log = logging.getLogger(__name__)

# ── Module-level defaults ──────────────────────────────────────────────────────

REGISTRY_COLUMNS      = ["id", "date", "title", "link", "guid"]
DEFAULT_EMBED_MODEL   = "text-embedding-3-small"
DEFAULT_EMBED_DIMS    = 1536
DEFAULT_POLL_INTERVAL = 30  # seconds


# ── Registry ──────────────────────────────────────────────────────────────────

def load_registry() -> pd.DataFrame:
    """Load the article registry, or return an empty frame if it does not exist.

    The registry is the ground truth for which articles are in the FAISS store.

    Returns:
        DataFrame with columns: id (int), date (str), title (str),
        link (str), guid (str).
    """
    if not FEEDS_REGISTRY_FILE.exists():
        log.info("Registry not found — starting fresh.")
        return pd.DataFrame(columns=REGISTRY_COLUMNS).astype({"id": int})

    df = pd.read_csv(FEEDS_REGISTRY_FILE, sep="\t", dtype={"id": int})
    log.info("Loaded registry: %d articles.", len(df))
    return df


def save_registry(registry: pd.DataFrame) -> None:
    """Write the registry as a TSV file.

    Uses an atomic write (tmp → rename) to prevent a half-written file
    if the process is interrupted between write and flush.

    Must be called AFTER store.save_local() so that a store write failure
    leaves the registry at its previous consistent state.
    """
    FEEDS_REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = FEEDS_REGISTRY_FILE.with_suffix(".tmp")
    registry.to_csv(tmp_path, sep="\t", index=False)
    tmp_path.replace(FEEDS_REGISTRY_FILE)
    log.info("Registry saved: %d articles total.", len(registry))


# ── Feed loading ──────────────────────────────────────────────────────────────

def load_all_feed_files() -> pd.DataFrame:
    """Load all top-level RAW_FEED_DIR/feeds*.txt files into a single DataFrame.

    Top-level only (no recursion into subdirectories).

    Deduplicates on guid (stable per-article identifier from the RSS feed).
    Adds a 'date' column (YYYY-MM-DD) derived from pubDate.

    Returns:
        DataFrame with columns: link, guid, type, id, sponsored, title,
        description, pubDate, date.

    Exits 0 if no feed files are found.
    """
    feed_files = sorted(RAW_FEED_DIR.glob("feeds*.txt"))
    if not feed_files:
        log.error("No feed files found in %s. Exiting.", RAW_FEED_DIR)
        sys.exit(0)

    raw_dfs  = [pd.read_csv(f, sep="\t") for f in feed_files]
    combined = pd.concat(raw_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["guid"])
    combined["date"] = pd.to_datetime(combined["pubDate"]).dt.strftime("%Y-%m-%d")

    log.info(
        "Loaded %d unique articles from %d file(s).",
        len(combined),
        len(feed_files),
    )
    return combined


def find_new_articles(all_df: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    """Return rows whose guid is not yet recorded in the registry.

    Exits 0 cleanly if there is nothing new to embed.
    """
    known_guids = set(registry["guid"]) if not registry.empty else set()
    new_df = all_df[~all_df["guid"].isin(known_guids)].copy()

    log.info(
        "%d total articles | %d already embedded | %d new",
        len(all_df),
        len(known_guids),
        len(new_df),
    )

    if new_df.empty:
        log.info("No new articles to embed. Exiting.")
        sys.exit(0)

    return new_df


def assign_ids(new_df: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    """Assign globally unique monotonic integer IDs to new articles.

    Continues from max(registry.id) + 1, or starts at 0 on the first run.
    IDs are never reused.
    """
    next_id = int(registry["id"].max()) + 1 if not registry.empty else 0
    new_df  = new_df.copy()
    new_df["id"] = range(next_id, next_id + len(new_df))
    log.info("Assigned IDs %d–%d.", next_id, next_id + len(new_df) - 1)
    return new_df


# ── Document building ─────────────────────────────────────────────────────────

def build_documents(new_df: pd.DataFrame) -> list[Document]:
    """Convert feed rows to LangChain Documents ready for batch embedding.

    Content format: "{date}: {title}: {description}"

    Each document carries:
        metadata["id"]    (int)  — docstore key in the FAISS store
        metadata["date"]  (str)  — YYYY-MM-DD
        metadata["title"] (str)  — article headline
        metadata["link"]  (str)  — article URL
        metadata["guid"]  (str)  — stable RSS article identifier
    """
    docs = []
    for _, row in new_df.iterrows():
        content = (
            f"{row['date']}: "
            f"{str(row['title']).strip()}: "
            f"{str(row['description']).strip()}"
        )
        docs.append(Document(
            page_content=content,
            metadata={
                "id":    int(row["id"]),
                "date":  str(row["date"]),
                "title": str(row["title"]),
                "link":  str(row["link"]),
                "guid":  str(row["guid"]),
            },
        ))

    log.info("Built %d documents for embedding.", len(docs))
    return docs


# ── Embedding batch ───────────────────────────────────────────────────────────

def run_embedding_batch(
    docs: list[Document],
    client: OpenAI,
    *,
    embed_model: str = DEFAULT_EMBED_MODEL,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
) -> list[tuple[str, list[float]]]:
    """Submit docs to the OpenAI Batch API and return (custom_id, embedding) pairs.

    Uses kitai.batch throughout:
        build_embedding_tasks → submit_batch_job → poll_until_complete
        → download_batch_results → parse_embedding_results

    custom_id in results == raw doc.metadata["id"] value (no prefix).

    Args:
        docs:          Documents to embed.
        client:        Initialised OpenAI client.
        embed_model:   Embedding model name (default: DEFAULT_EMBED_MODEL).
        poll_interval: Seconds between batch status polls (default: DEFAULT_POLL_INTERVAL).

    Raises:
        RuntimeError: If the batch job does not reach 'completed' status.
    """
    tasks  = build_embedding_tasks(docs, model=embed_model)
    job_id = submit_batch_job(client, tasks)
    log.info("Submitted embedding batch: %s (%d tasks)", job_id, len(tasks))

    statuses = poll_until_complete(client, [job_id], poll_interval_seconds=poll_interval)
    if statuses[job_id]["status"] != "completed":
        raise RuntimeError(
            f"Embedding batch {job_id} did not complete successfully "
            f"(status={statuses[job_id]['status']}). "
            "Check the OpenAI dashboard for details."
        )

    results = download_batch_results(client, job_id)
    pairs   = parse_embedding_results(results)

    log.info("Parsed %d embeddings from batch %s.", len(pairs), job_id)
    return pairs


def align_pairs_to_docs(
    pairs: list[tuple[str, list[float]]],
    docs: list[Document],
) -> tuple[list[tuple[str, list[float]]], list[Document]]:
    """Re-align (custom_id, embedding) pairs to match the docs list order.

    Drops any doc whose embedding is missing (API-level failure for that item).
    Those articles will be retried on the next run since they won't be in the
    registry.

    Args:
        pairs: List of (custom_id, embedding) from parse_embedding_results.
               custom_id is the raw doc.metadata["id"] value (int cast to str).
        docs:  Original document list in submission order.

    Returns:
        aligned_text_emb_pairs: List[(page_content, embedding)] — input for
            FAISS.add_embeddings or create_vectorstore.
        aligned_docs: Corresponding Document list (same order, failures excluded).
    """
    emb_by_id = {int(cid): emb for cid, emb in pairs}

    aligned_pairs: list[tuple[str, list[float]]] = []
    aligned_docs:  list[Document] = []
    dropped = 0

    for doc in docs:
        doc_id = doc.metadata["id"]
        if doc_id in emb_by_id:
            aligned_pairs.append((doc.page_content, emb_by_id[doc_id]))
            aligned_docs.append(doc)
        else:
            log.warning(
                "No embedding returned for doc id=%d — will retry next run.", doc_id
            )
            dropped += 1

    if dropped:
        log.warning("%d document(s) excluded due to missing embeddings.", dropped)

    log.info("Aligned %d document-embedding pairs.", len(aligned_docs))
    return aligned_pairs, aligned_docs


# ── Vectorstore init / update ─────────────────────────────────────────────────

def init_vectorstore(
    docs: list[Document],
    text_emb_pairs: list[tuple[str, list[float]]],
    embeddings_model: OpenAIEmbeddings,
) -> FAISS:
    """Build a new FAISS store from scratch (cold start path).

    Uses kitai.index.create_vectorstore which requires:
    - len(docs) == len(text_emb_pairs)
    - every doc.metadata["id"] is unique

    The embeddings_model is stored as the query encoder for future
    similarity_search("query string") calls. It is NOT called here.
    """
    embeddings_ndarray = np.array(
        [emb for _, emb in text_emb_pairs], dtype=np.float32
    )
    store = create_vectorstore(docs, embeddings_ndarray, embeddings_model)
    log.info("Initialised new vectorstore with %d vectors.", store.index.ntotal)
    return store


def update_vectorstore(
    text_emb_pairs: list[tuple[str, list[float]]],
    aligned_docs: list[Document],
    embeddings_model: OpenAIEmbeddings,
) -> FAISS:
    """Load the existing FAISS store and append new embeddings (incremental path).

    FAISS.add_embeddings grows the index in-place — no full rebuild needed.

    Integer metadata["id"] values are passed as docstore keys to stay
    consistent with the integer keys created by kitai.index.create_vectorstore
    at init time.
    """
    store  = FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings_model,
        allow_dangerous_deserialization=True,
    )
    before = store.index.ntotal

    store.add_embeddings(
        text_embeddings=text_emb_pairs,
        metadatas=[doc.metadata for doc in aligned_docs],
        ids=[doc.metadata["id"] for doc in aligned_docs],
    )

    log.info(
        "Updated vectorstore: %d -> %d vectors (+%d).",
        before,
        store.index.ntotal,
        store.index.ntotal - before,
    )
    return store
