"""Check or rebuild the EnBot Milvus knowledge collection."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from pymilvus import MilvusClient

from rag.knowledge_retriever import get_knowledge_retriever
from rag.vector_store import VectorStoreSettings


def client_for(settings: VectorStoreSettings) -> MilvusClient:
    kwargs = {"uri": settings.milvus_uri}
    if settings.milvus_token:
        kwargs["token"] = settings.milvus_token
    if settings.milvus_db_name:
        kwargs["db_name"] = settings.milvus_db_name
    return MilvusClient(**kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check or rebuild the EnBot Milvus collection."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only test the connection and show collection status.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm dropping and rebuilding the configured collection.",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    os.environ["VECTOR_STORE"] = "milvus"
    settings = VectorStoreSettings.from_env()
    try:
        client = client_for(settings)
        exists = bool(client.has_collection(settings.milvus_collection))
    except Exception as exc:
        print(f"[ERROR] Cannot connect to Milvus at {settings.milvus_uri}: {exc}")
        print("Start Docker Desktop and run scripts/09_start_milvus.ps1 first.")
        return 1

    print(f"Milvus URI: {settings.milvus_uri}")
    print(f"Database: {settings.milvus_db_name}")
    print(f"Collection: {settings.milvus_collection}")
    print(f"Collection exists: {exists}")

    if args.check:
        if exists:
            print(f"Stats: {client.get_collection_stats(settings.milvus_collection)}")
        return 0

    if not args.yes:
        print("Rebuild not started. Re-run with --yes after checking the settings.")
        return 2

    get_knowledge_retriever(force_rebuild=True, raise_on_error=True)
    client = client_for(settings)
    print(f"Rebuild complete: {client.get_collection_stats(settings.milvus_collection)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
