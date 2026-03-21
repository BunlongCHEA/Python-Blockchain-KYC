"""
Field extraction for Cambodian National ID Cards.
Strategy:
  1. Try MRZ lines first  (ICAO 9303 — always ASCII, always reliable)
  2. Fall back to label-based regex on full OCR text
"""
import re
from typing import Any, Dict, List


# ── MRZ parser (Cambodian ID — TD1 format: 3 lines × 30 chars) ───────────────

def _parse_mrz(texts: List[str]) -> Dict[str, str]:
    """
    Extract fields from MRZ lines if present.
    Cambodian National ID uses TD1 (3-line × 30 chars).
    Passport uses TD3 (2-line × 44 chars) — handled in extractor_passport.py.

    Line 1: IDKHM<id_number><<<<<<<<<<<<<<
    Line 2: YYMMDD<check><sex><expiry><check>KHM<<<<<<<<<<check>
    Line 3: LAST<<FIRST<MIDDLE<<<<<<<<<<<<
    """
    mrz_lines = []
    for t in texts:
        # Normalise: uppercase, replace common OCR mistakes
        clean = t.upper().replace(" ", "").replace("\n", "")
        # MRZ lines are 30 chars for TD1, 44 for TD3
        # Accept lines that are mostly alphanumeric + <
        if re.fullmatch(r"[A-Z0-9<\*]{28,44}", clean):
            mrz_lines.append(clean)

    if len(mrz_lines) < 2:
        return {}

    result = {}

    # ── Find TD1 triplet ──────────────────────────────────────────────────────
    for i in range(len(mrz_lines) - 1):
        l1 = mrz_lines[i]
        l2 = mrz_lines[i + 1]

        # Line 1: starts with ID or I + country code (e.g. IDKHM)
        if re.match(r"^ID[A-Z]{3}", l1):
            # ID number: chars 5–14 of line 1, strip trailing <
            id_raw = l1[5:14].replace("<", "").strip()
            if id_raw:
                result["id_number"] = id_raw

            # Line 2: YYMMDD + check + sex + expiry + check + nationality
            if len(l2) >= 30:
                dob_raw = l2[0:6]   # YYMMDD
                sex_raw = l2[7]
                exp_raw = l2[8:14]  # YYMMDD
                nat_raw = l2[15:18].replace("<", "")

                result["date_of_birth"] = _mrz_date(dob_raw, is_birth=True)
                result["expiry_date"]   = _mrz_date(exp_raw, is_birth=False)
                result["sex"]           = "M" if sex_raw == "M" else ("F" if sex_raw == "F" else "")
                if nat_raw:
                    result["nationality"] = _nat_code(nat_raw)

            # Line 3: LAST<<FIRST<MIDDLE (if exists)
            if i + 2 < len(mrz_lines):
                l3 = mrz_lines[i + 2]
                name_part = l3.split("<<", 1)
                result["last_name"]  = name_part[0].replace("<", " ").strip().title()
                result["first_name"] = name_part[1].replace("<", " ").strip().title() if len(name_part) > 1 else ""
            break

        # Line 1: TD3 passport-style (P<KHM...) — skip here, handled in passport extractor
        if re.match(r"^P[<A-Z][A-Z]{3}", l1):
            break

    return result


def _mrz_date(yymmdd: str, is_birth: bool) -> str:
    """Convert YYMMDD → DD/MM/YYYY. Birth years: 00-30 → 2000s, 31-99 → 1900s."""
    if len(yymmdd) != 6 or not yymmdd.isdigit():
        return ""
    yy, mm, dd = yymmdd[:2], yymmdd[2:4], yymmdd[4:6]
    if is_birth:
        yyyy = f"19{yy}" if int(yy) >= 31 else f"20{yy}"
    else:
        yyyy = f"20{yy}"
    return f"{dd}/{mm}/{yyyy}"


def _nat_code(code: str) -> str:
    _map = {"KHM": "Cambodian", "THA": "Thai", "VNM": "Vietnamese", "LAO": "Lao"}
    return _map.get(code.upper(), code)


# ── Label-based fallback (for non-MRZ / partially readable cards) ─────────────

def _label_extract(full: str) -> Dict[str, str]:
    def search(*patterns: str) -> str:
        for pat in patterns:
            m = re.search(pat, full, re.IGNORECASE | re.UNICODE)
            if m:
                return m.group(1).strip()
        return ""

    return {
        # ID number — standalone alphanumeric code
        "id_number": search(
            r"(?:ល\.|ID\s*No\.?|No\.?)\s*[:\-]?\s*([A-Z0-9]{6,20})",
            r"\b(ID[A-Z]{3}[A-Z0-9]{6,12})\b",
            r"\b([A-Z]{1,3}\d{7,12})\b",
        ),
        # Name — Khmer Unicode or Latin after label
        "first_name": search(
            r"(?:នាម|First\s*Name)\s*[:\-]?\s*([\u1780-\u17FF\s]{2,40})",
            r"(?:First\s*Name)\s*[:\-]?\s*([A-Za-z\s\-]{2,40})",
            # plain "MY NAME" style after colon
            r"name\s*[:\-]\s*([A-Za-z\s]{2,30})",
        ),
        "last_name": search(
            r"(?:គោត្ត|Family\s*Name|Last\s*Name)\s*[:\-]?\s*([\u1780-\u17FF\s]{2,40})",
            r"(?:(?:Family|Last)\s*Name)\s*[:\-]?\s*([A-Za-z\s\-]{2,40})",
        ),
        # Date — any DD.MM.YYYY or DD/MM/YYYY near DOB keywords
        "date_of_birth": search(
            r"(?:ថ្ងៃ[^\d]*|DOB|Date\s*of\s*Birth|Birth)\s*[:\-]?\s*"
            r"(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
            # fallback: any 8-digit date pattern in text
            r"(\d{2}[\.\/]\d{2}[\.\/]\d{4})",
        ),
        "expiry_date": search(
            r"(?:Exp(?:iry)?|Valid\s*(?:Until|Till)?)\s*[:\-]?\s*"
            r"(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
        ),
        "sex": search(
            r"(?:Sex|Gender|ភេទ|Sis)\s*[:\-]?\s*([MFmf\u1780-\u17FF]{1,10})"
        ),
        "nationality": search(
            r"(?:Nationality|ជាតិ)\s*[:\-]?\s*([\u1780-\u17FF A-Za-z]{3,30})"
        ),
    }


# ── Public entry point ────────────────────────────────────────────────────────

def extract_cambodian_id_fields(texts: List[str]) -> Dict[str, Any]:
    """
    Parse OCR strings into structured ID card fields.
    Tries MRZ first (most reliable), then falls back to label regex.
    """
    # Step 1: MRZ
    fields = _parse_mrz(texts)

    # Step 2: Fill any blanks with label-based extraction
    full   = "\n".join(texts)
    label  = _label_extract(full)

    merged: Dict[str, Any] = {}
    for key in ("id_number", "first_name", "last_name", "date_of_birth",
                "expiry_date", "sex", "nationality"):
        merged[key] = fields.get(key) or label.get(key, "")

    return merged


# Existing code for Tesseract OCR (for reference, not used in current Google Doc AI flow)

# """
# Field extraction for Cambodian National ID Cards.
# Handles both Khmer (Unicode block U+1780–U+17FF) and Latin labels.
# """

# def extract_cambodian_id_fields(texts: List[str]) -> Dict[str, Any]:
#     """
#     Parse a flat list of OCR strings into structured ID card fields.

#     Recognised fields
#     -----------------
#     id_number      - card serial / ID number
#     first_name     - given name (Khmer or Latin)
#     last_name      - family name (Khmer or Latin)
#     date_of_birth  - DD/MM/YYYY or similar
#     expiry_date    - document expiry
#     sex            - M / F
#     nationality    - e.g. Cambodian / ខ្មែរ
#     """
#     full = " ".join(texts)

#     def search(*patterns: str) -> str:
#         for pat in patterns:
#             m = re.search(pat, full, re.IGNORECASE | re.UNICODE)
#             if m:
#                 return m.group(1).strip()
#         return ""

#     return {
#         "id_number": search(
#             # Khmer label
#             r"(?:ល\.|ID\s*No\.?|No\.?)\s*[:\-]?\s*([A-Z0-9\u1780-\u17FF]{6,20})",
#             # Standalone code pattern
#             r"\b([A-Z]{1,2}\d{7,12})\b",
#         ),
#         "first_name": search(
#             r"(?:នាម|First\s*Name)\s*[:\-]?\s*([\u1780-\u17FF\s]{2,40})",
#             r"(?:First\s*Name)\s*[:\-]?\s*([A-Za-z\s\-]{2,40})",
#         ),
#         "last_name": search(
#             r"(?:គោត្ត|Family\s*Name|Last\s*Name)\s*[:\-]?\s*([\u1780-\u17FF\s]{2,40})",
#             r"(?:(?:Family|Last)\s*Name)\s*[:\-]?\s*([A-Za-z\s\-]{2,40})",
#         ),
#         "date_of_birth": search(
#             r"(?:ថ្ងៃ[^\d]*|DOB|Date\s*of\s*Birth)\s*[:\-]?\s*"
#             r"(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
#         ),
#         "expiry_date": search(
#             r"(?:Exp(?:iry)?|Valid\s*(?:Until|Till)?)\s*[:\-]?\s*"
#             r"(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
#         ),
#         "sex": search(
#             r"(?:Sex|Gender|ភេទ)\s*[:\-]?\s*([MFmf\u1780-\u17FF]{1,10})"
#         ),
#         "nationality": search(
#             r"(?:Nationality|ជាតិ)\s*[:\-]?\s*([\u1780-\u17FF A-Za-z]{3,30})"
#         ),
#     }