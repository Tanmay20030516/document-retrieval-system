import chromadb
import pandas as pd
from load_collections import get_collection
# from sentence_transformers import CrossEncoder
from utils import re_rank, cross_encoder_rerank, get_dataframe_to_display

pd.set_option("max_colwidth", None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
chroma_client = chromadb.PersistentClient(path=r'D:\\Coding\\Research-paper-retrieval-system\\vector_db')
df = pd.read_parquet(r"D:\\Coding\\Research-paper-retrieval-system\\app\\metadata.parquet")
collection1 = get_collection(
    name="distilbert_embedding_collection",
    client=chroma_client
)
print(collection1.name)

# collection2 = get_collection(
#     name="sbert_all-MiniLM-L6-v2_embedding_collection",
#     client=chroma_client
# )
# print(collection2)

results1 = collection1.query(
    query_texts=["approaches to learning sentence level embeddings"],  # Chroma will embed this for you
    n_results=10  # how many results to return
)
print(get_dataframe_to_display(results1['ids'][0], df))

# results2 = collection2.query(
#     query_texts=["approaches to learning sentence level embeddings"],  # Chroma will embed this for you
#     n_results=10  # how many results to return
# )
# print(get_dataframe_to_display(results2['ids'][0], df))


