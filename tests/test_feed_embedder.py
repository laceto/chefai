"""
tests/test_feed_embedder.py
Tests for chefai.feed_embedder.

Coverage targets
----------------
Registry:
    load_registry       — missing file, round-trip, atomic tmp cleanup
    save_registry       — written columns, no .tmp leftover

Feed loading:
    load_all_feed_files — no files → sys.exit(0), deduplication, date column
    find_new_articles   — empty registry, all known, partial
    assign_ids          — fresh start, incremental, contiguous IDs

Document building:
    build_documents     — count, content format, metadata keys/types

Pair alignment:
    align_pairs_to_docs — full match, missing embedding dropped, empty input

Embedding batch (mocked kitai.batch):
    run_embedding_batch — happy path, failed job → RuntimeError

Fixtures
--------
sample_feed_df  3-row DataFrame in feeds*.txt format (Italian recipes as data)
sample_docs     3 LangChain Documents derived from sample_feed_df
"""

import pandas as pd
import pytest
from langchain_core.documents import Document
from unittest.mock import MagicMock, patch

import chefai.feed_embedder as fe


# ── Sample data ───────────────────────────────────────────────────────────────
#
# Three Italian recipes used as synthetic feed articles.
# The guid and link are stable identifiers — same role as RSS guids.

_SAMPLE_ROWS = [
    {
        "title":       "Bagna Cauda",
        "guid":        "guid-bagna-cauda",
        "link":        "https://example.com/bagna-cauda",
        "description": "Piemontese garlic and anchovy sauce.",
        "pubDate":     "2026-01-01",
        "type":        "article",
        "sponsored":   False,
    },
    {
        "title":       "Bruschetta alla Romana",
        "guid":        "guid-bruschetta",
        "link":        "https://example.com/bruschetta",
        "description": "Toasted bread with tomatoes and basil.",
        "pubDate":     "2026-01-02",
        "type":        "article",
        "sponsored":   False,
    },
    {
        "title":       "Panzanella",
        "guid":        "guid-panzanella",
        "link":        "https://example.com/panzanella",
        "description": "Tuscan bread salad with summer vegetables.",
        "pubDate":     "2026-01-03",
        "type":        "article",
        "sponsored":   False,
    },
]


@pytest.fixture
def sample_feed_df() -> pd.DataFrame:
    """3-row DataFrame in feeds*.txt format, with derived `date` column."""
    df = pd.DataFrame(_SAMPLE_ROWS)
    df["date"] = pd.to_datetime(df["pubDate"]).dt.strftime("%Y-%m-%d")
    return df


@pytest.fixture
def sample_docs(sample_feed_df: pd.DataFrame) -> list[Document]:
    """3 Documents with ids 0–2, built from sample_feed_df."""
    df = sample_feed_df.copy()
    df["id"] = range(len(df))
    return fe.build_documents(df)


# ── Registry ──────────────────────────────────────────────────────────────────


def test_load_registry_missing_returns_empty_frame(tmp_path, monkeypatch):
    """load_registry returns an empty frame with correct schema when no file exists."""
    monkeypatch.setattr(fe, "FEEDS_REGISTRY_FILE", tmp_path / "registry.tsv")

    df = fe.load_registry()

    assert list(df.columns) == fe.REGISTRY_COLUMNS
    assert len(df) == 0
    assert df.dtypes["id"] == "int64"


def test_save_load_registry_roundtrip(tmp_path, monkeypatch):
    """save_registry then load_registry reproduces the same rows."""
    monkeypatch.setattr(fe, "FEEDS_REGISTRY_FILE", tmp_path / "registry.tsv")

    original = pd.DataFrame({
        "id":    [0, 1, 2],
        "date":  ["2026-01-01", "2026-01-02", "2026-01-03"],
        "title": ["Bagna Cauda", "Bruschetta alla Romana", "Panzanella"],
        "link":  ["https://example.com/a", "https://example.com/b", "https://example.com/c"],
        "guid":  ["guid-a", "guid-b", "guid-c"],
    })

    fe.save_registry(original)
    reloaded = fe.load_registry()

    pd.testing.assert_frame_equal(
        original.reset_index(drop=True),
        reloaded.reset_index(drop=True),
    )


def test_save_registry_no_tmp_leftover(tmp_path, monkeypatch):
    """The .tmp staging file is removed after a successful save."""
    registry_path = tmp_path / "registry.tsv"
    monkeypatch.setattr(fe, "FEEDS_REGISTRY_FILE", registry_path)

    fe.save_registry(pd.DataFrame(columns=fe.REGISTRY_COLUMNS))

    assert registry_path.exists()
    assert not registry_path.with_suffix(".tmp").exists()


# ── Feed loading ──────────────────────────────────────────────────────────────


def test_load_all_feed_files_no_files_exits(tmp_path, monkeypatch):
    """sys.exit(0) is called when no feeds*.txt files exist."""
    monkeypatch.setattr(fe, "RAW_FEED_DIR", tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        fe.load_all_feed_files()

    assert exc_info.value.code == 0


def test_load_all_feed_files_deduplicates(tmp_path, monkeypatch, sample_feed_df):
    """Duplicate guids across multiple files are reduced to one row each."""
    monkeypatch.setattr(fe, "RAW_FEED_DIR", tmp_path)

    # Write the same 3 articles into two separate feed files.
    cols_to_write = ["title", "guid", "link", "description", "pubDate", "type", "sponsored"]
    sample_feed_df[cols_to_write].to_csv(tmp_path / "feeds_a.txt", sep="\t", index=False)
    sample_feed_df[cols_to_write].to_csv(tmp_path / "feeds_b.txt", sep="\t", index=False)

    result = fe.load_all_feed_files()

    assert len(result) == 3  # 6 rows in, 3 unique guids out


def test_load_all_feed_files_date_column(tmp_path, monkeypatch, sample_feed_df):
    """The `date` column is derived from `pubDate` in YYYY-MM-DD format."""
    monkeypatch.setattr(fe, "RAW_FEED_DIR", tmp_path)

    cols_to_write = ["title", "guid", "link", "description", "pubDate", "type", "sponsored"]
    sample_feed_df[cols_to_write].to_csv(tmp_path / "feeds.txt", sep="\t", index=False)

    result = fe.load_all_feed_files()

    assert "date" in result.columns
    assert list(result["date"]) == ["2026-01-01", "2026-01-02", "2026-01-03"]


# ── Article filtering ─────────────────────────────────────────────────────────


def test_find_new_articles_empty_registry(sample_feed_df):
    """All articles are returned when the registry is empty."""
    empty = pd.DataFrame(columns=fe.REGISTRY_COLUMNS).astype({"id": int})
    result = fe.find_new_articles(sample_feed_df, empty)
    assert len(result) == 3


def test_find_new_articles_all_known_exits(sample_feed_df):
    """sys.exit(0) is raised when every guid is already registered."""
    registry = pd.DataFrame({
        "id":    [0, 1, 2],
        "date":  ["2026-01-01", "2026-01-02", "2026-01-03"],
        "title": sample_feed_df["title"].tolist(),
        "link":  sample_feed_df["link"].tolist(),
        "guid":  sample_feed_df["guid"].tolist(),
    })

    with pytest.raises(SystemExit) as exc_info:
        fe.find_new_articles(sample_feed_df, registry)

    assert exc_info.value.code == 0


def test_find_new_articles_partial(sample_feed_df):
    """Only the articles whose guids are absent from the registry are returned."""
    registry = pd.DataFrame({
        "id":    [0],
        "date":  ["2026-01-01"],
        "title": ["Bagna Cauda"],
        "link":  ["https://example.com/bagna-cauda"],
        "guid":  ["guid-bagna-cauda"],
    })

    result = fe.find_new_articles(sample_feed_df, registry)

    assert len(result) == 2
    assert set(result["guid"]) == {"guid-bruschetta", "guid-panzanella"}


# ── ID assignment ─────────────────────────────────────────────────────────────


def test_assign_ids_fresh_start(sample_feed_df):
    """IDs start at 0 when the registry is empty."""
    empty = pd.DataFrame(columns=fe.REGISTRY_COLUMNS).astype({"id": int})
    result = fe.assign_ids(sample_feed_df, empty)
    assert list(result["id"]) == [0, 1, 2]


def test_assign_ids_incremental(sample_feed_df):
    """IDs continue from max(registry.id) + 1."""
    registry = pd.DataFrame({"id": [0, 1, 2, 3, 4]})
    result = fe.assign_ids(sample_feed_df, registry)
    assert list(result["id"]) == [5, 6, 7]


def test_assign_ids_contiguous(sample_feed_df):
    """Assigned IDs form a contiguous range with no gaps."""
    empty = pd.DataFrame(columns=fe.REGISTRY_COLUMNS).astype({"id": int})
    result = fe.assign_ids(sample_feed_df, empty)
    ids = list(result["id"])
    assert ids == list(range(ids[0], ids[0] + len(ids)))


# ── Document building ─────────────────────────────────────────────────────────


def test_build_documents_count(sample_feed_df):
    """One Document is produced per feed row."""
    df = sample_feed_df.copy()
    df["id"] = range(len(df))
    docs = fe.build_documents(df)
    assert len(docs) == 3


def test_build_documents_content_format(sample_feed_df):
    """page_content is formatted as '{date}: {title}: {description}'."""
    df = sample_feed_df.copy()
    df["id"] = range(len(df))
    docs = fe.build_documents(df)

    row = df.iloc[0]
    expected = f"{row['date']}: {row['title']}: {row['description']}"
    assert docs[0].page_content == expected


def test_build_documents_metadata_keys(sample_feed_df):
    """Every Document carries the 5 required metadata keys with correct types."""
    df = sample_feed_df.copy()
    df["id"] = range(len(df))
    docs = fe.build_documents(df)

    for doc in docs:
        assert isinstance(doc.metadata["id"],    int)
        assert isinstance(doc.metadata["date"],  str)
        assert isinstance(doc.metadata["title"], str)
        assert isinstance(doc.metadata["link"],  str)
        assert isinstance(doc.metadata["guid"],  str)


# ── Pair alignment ────────────────────────────────────────────────────────────


def test_align_pairs_all_match(sample_docs):
    """All docs are aligned when every id has a matching embedding."""
    fake_emb = [0.1] * 8
    pairs = [(str(doc.metadata["id"]), fake_emb) for doc in sample_docs]

    aligned_pairs, aligned_docs = fe.align_pairs_to_docs(pairs, sample_docs)

    assert len(aligned_pairs) == 3
    assert len(aligned_docs) == 3
    assert [d.metadata["id"] for d in aligned_docs] == [0, 1, 2]


def test_align_pairs_missing_embedding_dropped(sample_docs):
    """A doc with no returned embedding is silently dropped; the rest are kept."""
    fake_emb = [0.1] * 8
    # Only supply embeddings for ids 0 and 2 — id 1 (Bruschetta) is missing.
    pairs = [("0", fake_emb), ("2", fake_emb)]

    aligned_pairs, aligned_docs = fe.align_pairs_to_docs(pairs, sample_docs)

    assert len(aligned_docs) == 2
    assert {d.metadata["id"] for d in aligned_docs} == {0, 2}


def test_align_pairs_empty_input():
    """Empty pairs and empty docs produce two empty lists."""
    aligned_pairs, aligned_docs = fe.align_pairs_to_docs([], [])
    assert aligned_pairs == []
    assert aligned_docs == []


# ── Embedding batch (mocked kitai.batch) ─────────────────────────────────────


@patch("chefai.feed_embedder.parse_embedding_results")
@patch("chefai.feed_embedder.download_batch_results")
@patch("chefai.feed_embedder.poll_until_complete")
@patch("chefai.feed_embedder.submit_batch_job")
@patch("chefai.feed_embedder.build_embedding_tasks")
def test_run_embedding_batch_success(
    mock_build,
    mock_submit,
    mock_poll,
    mock_download,
    mock_parse,
    sample_docs,
):
    """Happy path: tasks are built, job submitted, polled, parsed; pairs returned."""
    fake_emb = [0.1] * 8
    mock_build.return_value   = [{"custom_id": "0", "body": {}}]
    mock_submit.return_value  = "batch_test123"
    mock_poll.return_value    = {"batch_test123": {"status": "completed"}}
    mock_download.return_value = [{}]  # content irrelevant — parse is mocked
    mock_parse.return_value   = [("0", fake_emb), ("1", fake_emb), ("2", fake_emb)]

    client = MagicMock()
    pairs  = fe.run_embedding_batch(sample_docs, client)

    mock_submit.assert_called_once_with(client, mock_build.return_value)
    mock_poll.assert_called_once_with(
        client, ["batch_test123"], poll_interval_seconds=fe.DEFAULT_POLL_INTERVAL
    )
    assert len(pairs) == 3


@patch("chefai.feed_embedder.parse_embedding_results")
@patch("chefai.feed_embedder.download_batch_results")
@patch("chefai.feed_embedder.poll_until_complete")
@patch("chefai.feed_embedder.submit_batch_job")
@patch("chefai.feed_embedder.build_embedding_tasks")
def test_run_embedding_batch_failed_job_raises(
    mock_build,
    mock_submit,
    mock_poll,
    mock_download,
    mock_parse,
    sample_docs,
):
    """RuntimeError is raised when the batch job status is not 'completed'."""
    mock_build.return_value  = []
    mock_submit.return_value = "batch_fail999"
    mock_poll.return_value   = {"batch_fail999": {"status": "failed"}}

    with pytest.raises(RuntimeError, match="batch_fail999"):
        fe.run_embedding_batch(sample_docs, MagicMock())

    # download and parse must not be called after a failed job
    mock_download.assert_not_called()
    mock_parse.assert_not_called()
