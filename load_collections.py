from load_models import get_embed_fn


def get_collection(name: str, client):
    """
    Retrieves or creates a Chroma collection based on the provided name.

    Args:
        name (str): Name of the collection to retrieve or create.
        client: Chroma client to interact with.

    Returns:
        collection: The requested Chroma collection.

    Raises:
        ValueError: If an unsupported collection name is provided.
        Exception: If there is an issue retrieving or creating the collection.
    """
    try:
        if name not in ["distilbert_embedding_collection", "sbert_all-MiniLM-L6-v2_embedding_collection"]:
            raise ValueError(f"Unsupported collection name: {name}")

        collection = client.get_collection(
            name=name,
            # metadata={
            #     "hnsw:space": "cosine",
            #     "hnsw:batch_size": 200,
            # },
            embedding_function=get_embed_fn(name)
        )
        print(f"Fetched {name} collection")
        return collection

    except Exception as e:
        print(f"Error retrieving or creating collection {name}: {e}")
        raise  # Re-raise the exception after logging

