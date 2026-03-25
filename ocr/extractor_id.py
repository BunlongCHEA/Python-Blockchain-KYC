"""
Field extraction for Cambodian National ID Cards.
Strategy:
  1. Try MRZ lines first  (ICAO 9303 — always ASCII, always reliable)
  2. Fall back to label-based regex on full OCR text
"""
import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


# ── MRZ parser (Cambodian ID — TD1 format: 3 lines × 30 chars) ───────────────

def _split_mrz_candidates(texts: List[str]) -> List[str]:
    """
    Build a list of 30-char MRZ line candidates from OCR text.
    Handles the common case where OCR merges TD1 Line 1 + Line 2 (and possibly
    Line 3) into a single long string with spaces/noise between them.
    """
    mrz_lines = []
    for t in texts:
        for line in t.upper().split("\n"):
            clean = re.sub(r"[^A-Z0-9<]", "<", line.strip())
            clean = clean.replace(" ", "")

            # Skip lines that are too short
            if len(clean) < 28:
                continue

            # Count real (non-filler) alphanumeric characters
            alnum = len(re.sub(r"<", "", clean))

            # Normal single MRZ line (28-44 chars)
            if 28 <= len(clean) <= 44:
                if alnum >= 10:
                    mrz_lines.append(clean)
                continue

            # Merged lines: OCR glued Line1+Line2 into one string.
            # ONLY split if the line actually starts with "IDXXX" (TD1 Line 1 marker).
            if len(clean) >= 56 and re.match(r"^ID[A-Z]{3}", clean):
                logger.info("[MRZ] Found merged line (%d chars): %s", len(clean), clean)
                for start in range(0, len(clean), 30):
                    chunk = clean[start:start + 30]
                    if len(chunk) >= 28:
                        chunk = chunk.ljust(30, "<")
                        chunk_alnum = len(re.sub(r"<", "", chunk))
                        if chunk_alnum >= 8:
                            logger.info("[MRZ]   chunk[%d]: %s (alnum=%d)", start, chunk, chunk_alnum)
                            mrz_lines.append(chunk)

    logger.info("[MRZ] Candidates found: %d → %s", len(mrz_lines), mrz_lines)
    return mrz_lines


def _parse_mrz(texts: List[str]) -> Dict[str, str]:
    """
    Extract fields from MRZ lines if present.
    Cambodian National ID uses TD1 (3-line × 30 chars).

    TD1 layout (each line = 30 characters):
      Line 1: [0:2]=ID [2:5]=country [5:14]=doc_number [14]=check [15:30]=optional
      Line 2: [0:6]=DOB [6]=check [7]=sex [8:14]=expiry [14]=check [15:18]=nat [18:29]=optional [29]=check
      Line 3: LAST<<FIRST<MIDDLE<<<<...
    """
    mrz_lines = _split_mrz_candidates(texts)

    if len(mrz_lines) < 2:
        logger.info("[MRZ] Not enough MRZ lines (%d), falling back to label extraction", len(mrz_lines))
        return {}

    result = {}

    for i in range(len(mrz_lines) - 1):
        l1 = mrz_lines[i]
        l2 = mrz_lines[i + 1]

        if re.match(r"^ID[A-Z]{3}", l1):
            logger.info("[MRZ] Parsing L1: %s", l1)
            logger.info("[MRZ] Parsing L2: %s", l2)

            # ── ID number from Line 1 ──
            # Grab all alphanumeric chars after "IDKHM" until first '<'
            country = l1[2:5]
            id_match = re.match(r"^ID[A-Z]{3}([A-Z0-9]+)", l1)
            if id_match:
                result["id_number"] = f"ID{country}{id_match.group(1)}"
                logger.info("[MRZ]   id_number = %s", result["id_number"])

            # ── Line 2 fields ──
            if len(l2) >= 28:
                dob_raw = l2[0:6]
                sex_raw = l2[7]
                exp_raw = l2[8:14]
                nat_raw = l2[15:18].replace("<", "")

                logger.info("[MRZ]   L2 dob_raw=%s sex_raw=%s exp_raw=%s nat_raw=%s",
                            dob_raw, sex_raw, exp_raw, nat_raw)

                result["date_of_birth"] = _mrz_date(dob_raw, is_birth=True)
                result["expiry_date"]   = _mrz_date(exp_raw, is_birth=False)
                result["sex"]           = "M" if sex_raw == "M" else ("F" if sex_raw == "F" else "")
                if nat_raw:
                    result["nationality"] = _nat_code(nat_raw)

            # ── Line 3 ──
            if i + 2 < len(mrz_lines):
                l3 = mrz_lines[i + 2]
                logger.info("[MRZ] Parsing L3: %s", l3)
                name_part = l3.split("<<", 1)
                result["last_name"]  = name_part[0].replace("<", " ").strip().title()
                result["first_name"] = name_part[1].replace("<", " ").strip().title() if len(name_part) > 1 else ""
            break

        if re.match(r"^P[<A-Z][A-Z]{3}", l1):
            break

    logger.info("[MRZ] Final MRZ result: %s", result)
    return result


def _mrz_date(yymmdd: str, is_birth: bool) -> str:
    """Convert YYMMDD → YYYY-MM-DD. Birth years: YY>=31 → 19YY, YY<31 → 20YY."""
    if len(yymmdd) != 6 or not yymmdd.isdigit():
        return ""
    yy, mm, dd = yymmdd[:2], yymmdd[2:4], yymmdd[4:6]
    if is_birth:
        yyyy = f"19{yy}" if int(yy) >= 31 else f"20{yy}"
    else:
        yyyy = f"20{yy}"
    return f"{yyyy}-{mm}-{dd}"


def _nat_code(code: str) -> str:
    _map = {
        "KHM": "Cambodian", "THA": "Thai", "VNM": "Vietnamese",
        "LAO": "Lao", "MMR": "Myanmar", "SGP": "Singaporean",
        "MYS": "Malaysian", "CHN": "Chinese", "JPN": "Japanese",
        "USA": "American", "GBR": "British", "FRA": "French",
    }
    return _map.get(code.upper().replace("<", ""), code)


# ── Label-based fallback ─────────────────────────────────────────────────────
# ONLY matches ASCII/Latin patterns — Khmer digits (១២៣) are NOT captured.

def _label_extract(full: str) -> Dict[str, str]:
    def search(*patterns: str) -> str:
        for pat in patterns:
            m = re.search(pat, full, re.IGNORECASE | re.UNICODE)
            if m:
                val = m.group(1).strip()
                logger.info("[LABEL] Pattern %s matched: '%s'", pat[:40], val)
                return val
        return ""

    return {
        "id_number": search(
            r"(?:ល\.|ID\s*No\.?|No\.?)\s*[:\-]?\s*([A-Z0-9]{6,20})",
            r"\b(ID[A-Z]{3}[A-Z0-9]{6,12})\b",
            r"\b([A-Z]{1,3}\d{7,12})\b",
        ),
        "first_name": search(
            r"(?:First\s*Name)\s*[:\-]?\s*([A-Za-z\s\-]{2,40})",
            r"name\s*[:\-]\s*([A-Za-z\s]{2,30})",
        ),
        "last_name": search(
            r"(?:(?:Family|Last)\s*Name)\s*[:\-]?\s*([A-Za-z\s\-]{2,40})",
        ),
        # ONLY match ASCII digits for dates — no Khmer digit matching
        "date_of_birth": search(
            r"(?:DOB|Date\s*of\s*Birth|Birth)\s*[:\-]?\s*"
            r"(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
        ),
        "expiry_date": search(
            r"(?:Exp(?:iry)?|Valid\s*(?:Until|Till)?)\s*[:\-]?\s*"
            r"(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
        ),
        "sex": search(
            r"(?:Sex|Gender)\s*[:\-]?\s*([MFmf])"
        ),
        "nationality": search(
            r"(?:Nationality)\s*[:\-]?\s*([A-Za-z]{3,30})"
        ),
    }


# ── Public entry point ────────────────────────────────────────────────────────

def extract_cambodian_id_fields(texts: List[str]) -> Dict[str, Any]:
    """
    Parse OCR strings into structured ID card fields.
    Tries MRZ first (most reliable), then falls back to label regex.
    """
    logger.info("[EXTRACT] Input texts (%d entries):", len(texts))
    for idx, t in enumerate(texts):
        logger.info("[EXTRACT]   [%d] %s", idx, repr(t))

    # Step 1: MRZ
    fields = _parse_mrz(texts)

    # Step 2: Fill any blanks with label-based extraction
    full   = "\n".join(texts)
    label  = _label_extract(full)

    merged: Dict[str, Any] = {}
    for key in ("id_number", "first_name", "last_name", "date_of_birth",
                "expiry_date", "sex", "nationality"):
        mrz_val   = fields.get(key, "")
        label_val = label.get(key, "")
        merged[key] = mrz_val or label_val
        if mrz_val:
            logger.info("[MERGE] %s = '%s' (from MRZ)", key, mrz_val)
        elif label_val:
            logger.info("[MERGE] %s = '%s' (from label fallback)", key, label_val)
        else:
            logger.info("[MERGE] %s = '' (no match)", key)

    return merged