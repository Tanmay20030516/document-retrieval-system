from typing import Optional
import numpy as np
import numpy.typing as npt
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from chromadb.utils import embedding_functions

# Custom Embedding Function for DistilBERT
class CustomEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(
        self,
        model_name_or_checkpoint: str = "bert-base-uncased",
        tokenizer_checkpoint: str = "bert-base-uncased",
        cache_dir: Optional[str] = None
    ):
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name_or_checkpoint,
            output_attentions=True,
            output_hidden_states=True
        ).distilbert
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_checkpoint,
            cache_dir=cache_dir
        )

    @staticmethod
    def normalize(vector: npt.NDArray) -> npt.NDArray:
        """
        Normalizes a vector to unit length using L2 norm
        """
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def __call__(self, text_input: Documents) -> Embeddings:
        inputs = self.tokenizer(
            text_input,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        self.model.eval()
        all_embeds = torch.tensor([[0] * self.model.config.dim])
        with torch.inference_mode():
            outputs = self.model(**inputs)
            hidden_states = outputs.hidden_states # (L+1) * (batch_size, seq_len, 768)
            layer6, layer4, layer1 = hidden_states[6], hidden_states[4], hidden_states[1]
            mean_l6, mean_l4, mean_l1 = torch.mean(layer6, dim=1), torch.mean(layer4, dim=1), torch.mean(layer1, dim=1) # (bs, 768)
            op_embedding = (mean_l6+mean_l4+mean_l1)/3
            # cls_embed = output[:, 0, :] # (batch_size, 768)
            all_embeds = torch.concat((all_embeds, op_embedding), dim=0)
        embeddings = all_embeds[1:]
        return [e.tolist() for e in self.normalize(embeddings)]


# distilbert_embed_fn = CustomEmbeddingFunction(
#     model_name_or_checkpoint=r'D:\\Coding\\Research-paper-retrieval-system\\model_artifacts\\model_checkpoint\\checkpoint-30938',
#     tokenizer_checkpoint=r'D:\\Coding\\Research-paper-retrieval-system\\model_artifacts\\tokenizer_checkpoint',
# )

# Embedding function for S-BERT
# sbert_embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
#     model_name=r"D:\\Coding\\Research-paper-retrieval-system\\model_artifacts\\model_checkpoint\\encoder_model"
# )


def get_embed_fn(name: str):
    """
    Returns the embedding function for the collection name passed.
    :param name: `str` name of collection in ChromaDB
    :return: embedding function for that collection
    """
    if name == "distilbert_embedding_collection":
        fn = CustomEmbeddingFunction(
    model_name_or_checkpoint=r'D:\\Coding\\Research-paper-retrieval-system\\model_artifacts\\model_checkpoint\\checkpoint-30938',
    tokenizer_checkpoint=r'D:\\Coding\\Research-paper-retrieval-system\\model_artifacts\\tokenizer_checkpoint',
)
        print('Loaded DistilBERT embedding function')
        return fn
    elif name == "sbert_all-MiniLM-L6-v2_embedding_collection":
        fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=r"D:\\Coding\\Research-paper-retrieval-system\\model_artifacts\\model_checkpoint\\encoder_model"
)
        print('Loaded S-BERT embedding function')
        return fn
    else:
        raise ValueError(f"Unsupported collection name: {name}")