"""
tests/test_feed_embedder.py
Tests for chefai.feed_embedder (recipe corpus edition).

Coverage targets
----------------
Registry:
    load_registry       — missing file, round-trip, atomic tmp cleanup
    save_registry       — written columns, no .tmp leftover

Recipe loading:
    load_all_recipes    — missing corpus → sys.exit(0), recipe_id computed
    find_new_recipes    — empty registry, all known, partial
    assign_ids          — fresh start, incremental, contiguous IDs

Document building:
    build_documents     — count, content format, metadata keys/types

Pair alignment:
    align_pairs_to_docs — full match, missing embedding dropped, empty input

Embedding batch (mocked kitai.batch):
    run_embedding_batch — happy path, failed job → RuntimeError

Fixtures
--------
sample_recipe_df  3-row DataFrame in recipe corpus format (Italian recipes)
sample_docs       3 LangChain Documents derived from sample_recipe_df
"""

import json
import pandas as pd
import pytest
from langchain_core.documents import Document
from unittest.mock import MagicMock, patch

import chefai.feed_embedder as fe


# ── Sample data ───────────────────────────────────────────────────────────────
#
# Three Italian recipes drawn from the actual corpus schema.

_SAMPLE_RECIPES = [
    {
        "title":        "Bagna Cauda",
        "source_file":  "antipasti-di-mare",
        "ingredients":  ["9 ACCIUGHE SOTTO SALE", "3 SPICCHI DI AGLIO", "3 DL DI OLIO"],
        "instructions": ["Sbucciate l'aglio. Cuocete per 10 minuti."],
        "pages":        [33],
        "difficulty":   "*",
    },
    {
        "title":        "Bruschetta alla Romana",
        "source_file":  "antipasti-dell-entroterra",
        "ingredients":  ["PANE CASERECCIO", "2 POMODORI", "BASILICO", "OLIO"],
        "instructions": ["Tostate il pane. Farcite con pomodoro e basilico."],
        "pages":        [12],
        "difficulty":   "*",
    },
    {
        "title":        "Panzanella",
        "source_file":  "antipasti-dell-entroterra",
        "ingredients":  ["PANE RAFFERMO", "POMODORI", "CIPOLLA", "ACETO"],
        "instructions": ["Ammollate il pane. Unite le verdure e condite."],
        "pages":        [14],
        "difficulty":   "*",
    },
]

_EXPECTED_RECIPE_IDS = [
    "antipasti-di-mare::Bagna Cauda",
    "antipasti-dell-entroterra::Bruschetta alla Romana",
    "antipasti-dell-entroterra::Panzanella",
]


@pytest.fixture
def sample_recipe_df() -> pd.DataFrame:
    """3-row DataFrame in recipe corpus format, with recipe_id column."""
    df = pd.DataFrame(_SAMPLE_RECIPES)
    df["recipe_id"] = df["source_file"] + "::" + df["title"]
    return df


@pytest.fixture
def sample_docs(sample_recipe_df: pd.DataFrame) -> list[Document]:
    """3 Documents with ids 0–2, built from sample_recipe_df."""
    df = sample_recipe_df.copy()
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
        "id":          [0, 1, 2],
        "recipe_id":   _EXPECTED_RECIPE_IDS,
        "title":       ["Bagna Cauda", "Bruschetta alla Romana", "Panzanella"],
        "source_file": ["antipasti-di-mare", "antipasti-dell-entroterra",
                        "antipasti-dell-entroterra"],
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


# ── Recipe loading ────────────────────────────────────────────────────────────


def test_load_all_recipes_missing_corpus_exits(tmp_path, monkeypatch):
    """sys.exit(0) is called when the corpus JSON file does not exist."""
    monkeypatch.setattr(fe, "RECIPES_CORPUS_FILE", tmp_path / "missing.json")

    with pytest.raises(SystemExit) as exc_info:
        fe.load_all_recipes()

    assert exc_info.value.code == 0


def test_load_all_recipes_computes_recipe_id(tmp_path, monkeypatch):
    """recipe_id is computed as '{source_file}::{title}'."""
    corpus_path = tmp_path / "recipes.json"
    corpus_path.write_text(json.dumps(_SAMPLE_RECIPES), encoding="utf-8")
    monkeypatch.setattr(fe, "RECIPES_CORPUS_FILE", corpus_path)

    result = fe.load_all_recipes()

    assert list(result["recipe_id"]) == _EXPECTED_RECIPE_IDS


def test_load_all_recipes_count(tmp_path, monkeypatch):
    """All records from the JSON file are loaded."""
    corpus_path = tmp_path / "recipes.json"
    corpus_path.write_text(json.dumps(_SAMPLE_RECIPES), encoding="utf-8")
    monkeypatch.setattr(fe, "RECIPES_CORPUS_FILE", corpus_path)

    result = fe.load_all_recipes()

    assert len(result) == 3


# ── Recipe filtering ──────────────────────────────────────────────────────────


def test_find_new_recipes_empty_registry(sample_recipe_df):
    """All recipes are returned when the registry is empty."""
    empty = pd.DataFrame(columns=fe.REGISTRY_COLUMNS).astype({"id": int})
    result = fe.find_new_recipes(sample_recipe_df, empty)
    assert len(result) == 3


def test_find_new_recipes_all_known_exits(sample_recipe_df):
    """sys.exit(0) is raised when every recipe_id is already registered."""
    registry = pd.DataFrame({
        "id":          [0, 1, 2],
        "recipe_id":   _EXPECTED_RECIPE_IDS,
        "title":       sample_recipe_df["title"].tolist(),
        "source_file": sample_recipe_df["source_file"].tolist(),
    })

    with pytest.raises(SystemExit) as exc_info:
        fe.find_new_recipes(sample_recipe_df, registry)

    assert exc_info.value.code == 0


def test_find_new_recipes_partial(sample_recipe_df):
    """Only recipes whose recipe_id is absent from the registry are returned."""
    registry = pd.DataFrame({
        "id":          [0],
        "recipe_id":   [_EXPECTED_RECIPE_IDS[0]],
        "title":       ["Bagna Cauda"],
        "source_file": ["antipasti-di-mare"],
    })

    result = fe.find_new_recipes(sample_recipe_df, registry)

    assert len(result) == 2
    assert set(result["recipe_id"]) == {
        "antipasti-dell-entroterra::Bruschetta alla Romana",
        "antipasti-dell-entroterra::Panzanella",
    }


# ── ID assignment ─────────────────────────────────────────────────────────────


def test_assign_ids_fresh_start(sample_recipe_df):
    """IDs start at 0 when the registry is empty."""
    empty = pd.DataFrame(columns=fe.REGISTRY_COLUMNS).astype({"id": int})
    result = fe.assign_ids(sample_recipe_df, empty)
    assert list(result["id"]) == [0, 1, 2]


def test_assign_ids_incremental(sample_recipe_df):
    """IDs continue from max(registry.id) + 1."""
    registry = pd.DataFrame({"id": [0, 1, 2, 3, 4]})
    result = fe.assign_ids(sample_recipe_df, registry)
    assert list(result["id"]) == [5, 6, 7]


def test_assign_ids_contiguous(sample_recipe_df):
    """Assigned IDs form a contiguous range with no gaps."""
    empty = pd.DataFrame(columns=fe.REGISTRY_COLUMNS).astype({"id": int})
    result = fe.assign_ids(sample_recipe_df, empty)
    ids = list(result["id"])
    assert ids == list(range(ids[0], ids[0] + len(ids)))


# ── Document building ─────────────────────────────────────────────────────────


def test_build_documents_count(sample_recipe_df):
    """One Document is produced per recipe row."""
    df = sample_recipe_df.copy()
    df["id"] = range(len(df))
    docs = fe.build_documents(df)
    assert len(docs) == 3


def test_build_documents_content_format(sample_recipe_df):
    """page_content is '{title}. Ingredienti: {ing}. Preparazione: {inst}'."""
    df = sample_recipe_df.copy()
    df["id"] = range(len(df))
    docs = fe.build_documents(df)

    row = df.iloc[0]
    ingredients  = "; ".join(str(i) for i in row["ingredients"])
    instructions = " ".join(str(s) for s in row["instructions"])
    expected = f"{row['title']}. Ingredienti: {ingredients}. Preparazione: {instructions}"
    assert docs[0].page_content == expected


def test_build_documents_metadata_keys(sample_recipe_df):
    """Every Document carries the 4 required metadata keys with correct types."""
    df = sample_recipe_df.copy()
    df["id"] = range(len(df))
    docs = fe.build_documents(df)

    for doc in docs:
        assert isinstance(doc.metadata["id"],          int)
        assert isinstance(doc.metadata["recipe_id"],   str)
        assert isinstance(doc.metadata["title"],       str)
        assert isinstance(doc.metadata["source_file"], str)


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
    mock_build.return_value    = [{"custom_id": "0", "body": {}}]
    mock_submit.return_value   = "batch_test123"
    mock_poll.return_value     = ["batch_test123"]
    mock_download.return_value = [{}]  # content irrelevant — parse is mocked
    mock_parse.return_value    = [("0", fake_emb), ("1", fake_emb), ("2", fake_emb)]

    client = MagicMock()
    pairs  = fe.run_embedding_batch(sample_docs, client)

    mock_submit.assert_called_once_with(
        client, mock_build.return_value, metadata={"description": "recipes_embed"}
    )
    mock_poll.assert_called_once_with(
        client, ["batch_test123"], poll_interval=fe.DEFAULT_POLL_INTERVAL
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
    mock_poll.return_value   = []  # job_id not in completed list → failed

    with pytest.raises(RuntimeError, match="batch_fail999"):
        fe.run_embedding_batch(sample_docs, MagicMock())

    # download and parse must not be called after a failed job
    mock_download.assert_not_called()
    mock_parse.assert_not_called()
