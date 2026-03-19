"""
Field extraction for Cambodian National ID Cards.
Handles both Khmer (Unicode block U+1780–U+17FF) and Latin labels.
"""
import re
from typing import Any, Dict, List


def extract_cambodian_id_fields(texts: List[str]) -> Dict[str, Any]:
    """
    Parse a flat list of OCR strings into structured ID card fields.

    Recognised fields
    -----------------
    id_number      - card serial / ID number
    first_name     - given name (Khmer or Latin)
    last_name      - family name (Khmer or Latin)
    date_of_birth  - DD/MM/YYYY or similar
    expiry_date    - document expiry
    sex            - M / F
    nationality    - e.g. Cambodian / ខ្មែរ
    """
    full = " ".join(texts)

    def search(*patterns: str) -> str:
        for pat in patterns:
            m = re.search(pat, full, re.IGNORECASE | re.UNICODE)
            if m:
                return m.group(1).strip()
        return ""

    return {
        "id_number": search(
            # Khmer label
            r"(?:ល\.|ID\s*No\.?|No\.?)\s*[:\-]?\s*([A-Z0-9\u1780-\u17FF]{6,20})",
            # Standalone code pattern
            r"\b([A-Z]{1,2}\d{7,12})\b",
        ),
        "first_name": search(
            r"(?:នាម|First\s*Name)\s*[:\-]?\s*([\u1780-\u17FF\s]{2,40})",
            r"(?:First\s*Name)\s*[:\-]?\s*([A-Za-z\s\-]{2,40})",
        ),
        "last_name": search(
            r"(?:គោត្ត|Family\s*Name|Last\s*Name)\s*[:\-]?\s*([\u1780-\u17FF\s]{2,40})",
            r"(?:(?:Family|Last)\s*Name)\s*[:\-]?\s*([A-Za-z\s\-]{2,40})",
        ),
        "date_of_birth": search(
            r"(?:ថ្ងៃ[^\d]*|DOB|Date\s*of\s*Birth)\s*[:\-]?\s*"
            r"(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
        ),
        "expiry_date": search(
            r"(?:Exp(?:iry)?|Valid\s*(?:Until|Till)?)\s*[:\-]?\s*"
            r"(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
        ),
        "sex": search(
            r"(?:Sex|Gender|ភេទ)\s*[:\-]?\s*([MFmf\u1780-\u17FF]{1,10})"
        ),
        "nationality": search(
            r"(?:Nationality|ជាតិ)\s*[:\-]?\s*([\u1780-\u17FF A-Za-z]{3,30})"
        ),
    }