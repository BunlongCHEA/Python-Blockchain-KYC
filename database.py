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
            SELECT first_name, last_name, date_of_birth,
                   id_number, id_expiry_date
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

        checks = {
            "first_name": norm(row["first_name"])
                == norm(extracted.get("first_name", "")),
            "last_name": norm(row["last_name"])
                == norm(extracted.get("last_name", "")),
            "date_of_birth": norm(str(row["date_of_birth"]))
                == norm(extracted.get("date_of_birth", "")),
            "id_number": norm(row["id_number"])
                == norm(
                    extracted.get("id_number", extracted.get("passport_number", ""))
                ),
        }
        score = round(sum(checks.values()) / len(checks), 4)
        return {"db_found": True, "matched_fields": checks, "match_score": score}

    except Exception as exc:
        logger.warning("DB match error: %s", exc)
        return {
            "db_found": False,
            "matched_fields": {},
            "match_score": 0.0,
            "error": str(exc),
        }