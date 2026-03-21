"""
PostgreSQL connection and KYC record lookup helpers.
"""
import re
import logging

import psycopg2
from psycopg2.extras import RealDictCursor

from config import settings

logger = logging.getLogger(__name__)


def get_connection():
    """Open and return a new psycopg2 connection."""
    return psycopg2.connect(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=settings.POSTGRES_DB,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
    )


def match_fields_with_db(customer_id: str, extracted: dict) -> dict:
    """
    Compare OCR-extracted document fields against the stored KYC record
    in PostgreSQL.

    Returns a dict with:
      - db_found       : bool
      - matched_fields : {field: bool}
      - match_score    : float 0-1
    """
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """
            SELECT first_name, last_name, date_of_birth,id_expiry_date
            FROM   kyc_records
            WHERE  customer_id = %s
            LIMIT  1
            """,
            (customer_id,),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()

        if not row:
            return {"db_found": False, "matched_fields": {}, "match_score": 0.0}

        def norm(s: str) -> str:
            return re.sub(r"\s+", " ", str(s or "").lower().strip())
        
        def norm_date(s: str) -> str:
            """Normalise date: remove separators, lowercase → comparable string."""
            return re.sub(r"[\s/\-\.]", "", str(s or "").lower().strip())

        checks = {
            "first_name": norm(row["first_name"])
                == norm(extracted.get("first_name", "")),
            "last_name": norm(row["last_name"])
                == norm(extracted.get("last_name", "")),
            # Date — normalise separators before comparing (DB: YYYY-MM-DD, OCR: DD/MM/YYYY)
            "date_of_birth": _date_match(
                str(row["date_of_birth"]),
                extracted.get("date_of_birth", ""),
            ),
        }
        score = round(sum(checks.values()) / len(checks), 4)
        return {
            "db_found": True, 
            "matched_fields": checks,
            "match_score": score
        }

    except Exception as exc:
        logger.warning("DB match error: %s", exc)
        return {
            "db_found": False,
            "matched_fields": {},
            "match_score": 0.0,
            "error": str(exc),
        }
        
def _date_match(db_date: str, ocr_date: str) -> bool:
    """
    Compare dates flexibly:
    DB stores  : 1988-08-08  (YYYY-MM-DD)
    OCR returns: 08/08/1988  (DD/MM/YYYY) or 08.08.1988
    → extract digits only and compare as sets of (dd, mm, yyyy)
    """
    if not db_date or not ocr_date:
        return False

    # Extract digit groups
    db_parts  = re.findall(r"\d+", db_date)   # ['1988','08','08']
    ocr_parts = re.findall(r"\d+", ocr_date)  # ['08','08','1988']

    if len(db_parts) < 3 or len(ocr_parts) < 3:
        return False

    # Normalise to sets of zero-padded values
    db_set  = {p.zfill(2) for p in db_parts}
    ocr_set = {p.zfill(2) for p in ocr_parts}

    # All 3 digit groups must appear in both (day, month, year)
    return db_set == ocr_set