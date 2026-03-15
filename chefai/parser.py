import json
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Recipe:
    title: str = ""
    prep_time: str = "N/A"
    cook_time: str = "N/A"
    difficulty: str = "N/A"
    servings: Optional[str] = None
    ingredients: List[str] = field(default_factory=list)
    instructions: List[str] = field(default_factory=list)
    pages: List[int] = field(default_factory=list)
    raw_text: str = ""
    confidence: float = 1.0  # 0.0–1.0, scende se parsing dubbioso


def normalize_text(text: str) -> str:
    """Pulizia aggressiva ma controllata – prepara per parsing"""
    text = re.sub(r'\s+', ' ', text.strip())
    text = text.replace('’', "'").replace('`', "'").replace('„', '"').replace('“', '"')
    # Fix spazi tra lettere maiuscole (tipico artefatto PDF)
    text = re.sub(r'([A-ZÀ-Ù])\s+([A-ZÀ-Ù])', r'\1\2', text)
    return text


def is_likely_title(line: str) -> bool:
    line = line.strip()
    if len(line) < 8 or len(line) > 80:
        return False

    upper_ratio = sum(c.isupper() for c in line) / max(len(line), 1)
    has_space = ' ' in line
    not_section_header = not any(
        kw.lower() in line.lower() for kw in ["ingredienti", "preparazione", "cottura", "procedimento", "difficoltà"]
    )

    return upper_ratio >= 0.7 and has_space and not_section_header


def is_continuation(page_text: str) -> bool:
    """Più segnali per decidere se è continuazione"""
    page_text = normalize_text(page_text)
    if not page_text:
        return True

    first_line = page_text.split('\n', 1)[0].strip()

    if is_likely_title(first_line):
        return False

    # Segnali forti di continuazione
    continuation_patterns = [
        r'^\s*(?:e\s+|poi\s+|quindi\s+|inoltre\s+|aggiungere\s+|mescolare\s+|fate\s+|cuocere\s+)',
        r'^\s*(?:il|la|lo|i|gli|le|un|una|dei|delle|con|in|su|per|di|da|a|al|dal)\b',
        r'^\s*[a-zàèìòù]',
        r'^\s*(?:Togliere|Eliminare|Scolare|Filtrare|Versare|Disporre|Cospargere|Infornare)',
    ]

    return any(re.match(p, first_line, re.IGNORECASE) for p in continuation_patterns)


def extract_key_value_block(text: str, keyword: str, default: str = "N/A") -> str:
    pat = rf'(?i){keyword}\s*[:;]?\s*([^,\n;]*?)(?:,|\n\n|$)'
    m = re.search(pat, text)
    return normalize_text(m.group(1)) if m else default


def parse_ingredients(text: str) -> List[str]:
    """Parsing ingredienti più intelligente – cerca quantità + unità + alimento"""
    block_match = re.search(
        r'(?s)(?:INGREDIENTI|Ingredienti)\s*(?:PER\s*\d+\s*(?:PERSONE|PERS\.))?[:\s-]*(.*?)(?=\n{2,}|PROCEDIMENTO|PREPARAZIONE|Si\s|Mettere|Lavat|Fate|Cuoc|$)',
        text, re.IGNORECASE
    )
    if not block_match:
        return []

    block = normalize_text(block_match.group(1))
    lines = [l.strip() for l in block.split('\n') if l.strip()]

    ingredients = []
    current = ""

    for line in lines:
        # Nuova riga ingrediente se inizia con numero, frazione o maiuscola alimento
        if re.match(r'^\d+[,\.]?\d*|\d+/\d+|^[A-ZÀ-Ù]', line):
            if current:
                ingredients.append(current.strip())
            current = line
        elif current:
            current += " " + line
        else:
            current = line

    if current:
        ingredients.append(current.strip())

    # Pulizia finale
    return [re.sub(r'\s+', ' ', ing).strip() for ing in ingredients if len(ing) > 3]


def split_instructions(text: str) -> List[str]:
    """Separa istruzioni in passi logici – più preciso di split semplice"""
    text = re.sub(r'\n\s*\n+', '\n', text)  # normalizza paragrafi
    sentences = re.split(r'(?<=[.;!?])\s+(?=[A-Z]|[A-Z].*?:|\d+\.)', text)

    steps = []
    for s in sentences:
        s = normalize_text(s)
        if len(s) > 12 and not s.lower().startswith(("per ", "con ", "sale e pepe")):
            steps.append(s)

    return steps


def extract_recipes_from_pages(page_texts: Dict[int, str]) -> List[Dict[str, Any]]:
    recipes: List[Recipe] = []
    current: Optional[Recipe] = None
    accumulated = ""

    for page_num in sorted(page_texts.keys()):
        page_content = normalize_text(page_texts[page_num])
        if not page_content:
            if current:
                current.raw_text += "\n\n[PAGINA VUOTA]"
            continue

        lines = [l.strip() for l in page_content.split('\n') if l.strip()]
        if not lines:
            continue

        first_line = lines[0]

        if is_likely_title(first_line):
            if current:
                current.raw_text = accumulated.strip()
                recipes.append(current)

            current = Recipe(
                title=first_line,
                pages=[page_num],
                raw_text=page_content
            )
            accumulated = page_content
        else:
            if current is None:
                logger.warning(f"Contenuto orfano pagina {page_num}")
                current = Recipe(title="Senza titolo", pages=[page_num], raw_text=page_content, confidence=0.6)
                accumulated = page_content
            else:
                current.pages.append(page_num)
                accumulated += "\n\n--- pag. {} ---\n{}".format(page_num, page_content)

    if current:
        current.raw_text = accumulated.strip()
        recipes.append(current)

    # Parsing strutturato
    structured = []
    for r in recipes:
        content = r.raw_text

        r.prep_time    = extract_key_value_block(content, "PREPARAZIONE")
        r.cook_time    = extract_key_value_block(content, "COTTURA")
        r.difficulty   = extract_key_value_block(content, r"DIFFICOLT[ÀA]")
        r.servings     = extract_key_value_block(content, r"(?:PER|per)\s*\d+", None)

        r.ingredients  = parse_ingredients(content)
        instr_start    = content.find("INGREDIENTI") + 100 if "INGREDIENTI" in content.upper() else 0
        instr_text     = content[instr_start:]
        r.instructions = split_instructions(instr_text)

        # Penalità fiducia se manca roba importante
        if not r.ingredients:
            r.confidence *= 0.6
        if len(r.instructions) < 2:
            r.confidence *= 0.7

        structured.append(asdict(r))

    return structured


# ---------------------------------------------------
#  Integrazione nella funzione principale (solo snippet)
# ---------------------------------------------------
def extract_recipes_from_pdf(pdf_path: str | Path, output_json: str = "ricette.json") -> None:
    # ... (stesso codice di prima per estrarre page_texts con fitz)

    page_texts = {}
    doc = fitz.open(pdf_path)
    for i in tqdm(range(len(doc)), desc="Estrazione testo"):
        page_texts[i+1] = doc[i].get_text("text")
    doc.close()

    recipes = extract_recipes_from_pages(page_texts)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(recipes, f, ensure_ascii=False, indent=2)

    logger.info(f"Estratte {len(recipes)} ricette (qualità media: {sum(r['confidence'] for r in recipes)/len(recipes):.2f})")