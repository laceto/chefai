"""Build the full recipe corpus from all processed markdown section files.

Usage
-----
Run from the project root with the virtual environment active::

    python scripts/build_corpus.py
    python scripts/build_corpus.py --no-raw-text
    python scripts/build_corpus.py --output data/my_corpus.json
    python scripts/build_corpus.py --processed-dir path/to/other/dir

Each ``*.md`` file in ``data/processed/`` is parsed with
:func:`chefai.extractor.parse_preparazioni_md`.  A ``source_file`` field
(the stem of the source path, e.g. ``"antipasti-di-mare"``) is stamped onto
every recipe dict so the corpus is self-describing.  The combined list is
written atomically to ``data/recipes.json`` via
:func:`chefai.extractor.export_corpus_to_json`.

A single malformed source file does not abort the build — its error is
logged and the remaining files are still processed.
"""
import argparse
import logging
from pathlib import Path

from chefai.extractor import export_corpus_to_json, parse_preparazioni_md

logger = logging.getLogger(__name__)


def build_corpus(
    processed_dir: Path,
    output_path: Path,
    include_raw_text: bool = True,
) -> list[dict]:
    """Parse all ``.md`` files in *processed_dir* and write a single corpus JSON.

    **Args:**
    - ``processed_dir`` — directory containing preprocessed section ``.md`` files.
    - ``output_path`` — destination for the combined ``recipes.json``.
    - ``include_raw_text`` — passed through to :func:`export_corpus_to_json`.

    **Returns** the combined list of recipe dicts (including ``source_file`` stamps).

    **Failure mode:** individual file errors are logged and skipped; the
    function still writes (and returns) whatever was successfully parsed.
    """
    md_files = sorted(processed_dir.glob("*.md"))
    if not md_files:
        logger.warning("No .md files found in %s", processed_dir)
        return []

    all_recipes: list[dict] = []
    for md_path in md_files:
        try:
            recipes = parse_preparazioni_md(md_path)
            source_stem = md_path.stem
            for r in recipes:
                r["source_file"] = source_stem
            all_recipes.extend(recipes)
            logger.info("  %s → %d recipes", md_path.name, len(recipes))
        except Exception as exc:
            logger.error("Failed to parse %s: %s", md_path.name, exc)

    export_corpus_to_json(all_recipes, output_path, include_raw_text=include_raw_text)
    logger.info("Corpus complete: %d total recipes → %s", len(all_recipes), output_path)
    return all_recipes


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/recipes.json"),
        metavar="PATH",
        help="Destination path for the JSON corpus (default: data/recipes.json)",
    )
    parser.add_argument(
        "--no-raw-text",
        action="store_true",
        help="Omit raw_text fields to reduce file size",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        metavar="DIR",
        help="Directory containing processed .md files (default: data/processed)",
    )
    args = parser.parse_args()
    build_corpus(
        processed_dir=args.processed_dir,
        output_path=args.output,
        include_raw_text=not args.no_raw_text,
    )


if __name__ == "__main__":
    main()
