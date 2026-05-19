"""Smoke test: verifica conectividad con Postgres y Neo4j del docker-compose."""

from __future__ import annotations

import sys

import psycopg
from dotenv import load_dotenv
from neo4j import GraphDatabase

from memgpt.config import get_settings


def check_postgres(dsn: str) -> bool:
    try:
        with psycopg.connect(dsn, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        print(f"[postgres] OK ({dsn})")
        return True
    except Exception as exc:
        print(f"[postgres] FAILED ({dsn}): {exc}")
        return False


def check_neo4j(uri: str, user: str, password: str) -> bool:
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        driver.close()
        print(f"[neo4j] OK ({uri})")
        return True
    except Exception as exc:
        print(f"[neo4j] FAILED ({uri}): {exc}")
        return False


def main() -> int:
    load_dotenv()
    settings = get_settings()

    ok = True
    ok &= check_postgres(settings.postgres_dsn)
    ok &= check_neo4j(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)

    if ok:
        print("OK")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
