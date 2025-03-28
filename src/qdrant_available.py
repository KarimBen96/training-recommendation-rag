"""
Script to list all available Qdrant collections in a given directory.
"""

import os
from qdrant_client import QdrantClient
import argparse


def list_collections(path):
    """List all collections in the specified Qdrant directory."""
    client = None

    if not os.path.exists(path):
        print(f"Error: Directory '{path}' does not exist!")
        return

    print(f"Checking for Qdrant collections in: {path}")

    try:
        client = QdrantClient(path=path)
        collections = client.get_collections()

        if not collections.collections:
            print(f"No collections found in {path}")
        else:
            print(f"Found {len(collections.collections)} collections:")
            for collection in collections.collections:
                print(f"  - {collection.name}")

    except Exception as e:
        print(f"Error accessing Qdrant at {path}: {e}")

        # List directories to help diagnose
        print("\nDirectory contents:")
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                print(f"  - {item}/ (directory)")
            else:
                print(f"  - {item} (file)")

    finally:
        # Ensure the client is properly closed even if exceptions occur
        if client is not None:
            try:
                del client
                print("Qdrant client connection closed.")
            except Exception as close_error:
                print(f"Warning: Error while closing client: {close_error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List Qdrant collections")
    parser.add_argument(
        "--path", default="db/qdrant_data", help="Path to Qdrant data directory"
    )
    args = parser.parse_args()

    list_collections(args.path)
