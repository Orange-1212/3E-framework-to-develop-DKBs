# rag/embeddings.py
# Embedding 统一出口（Q1 用 A）
# 本地 sentence-transformers，FAISS 友好，无 deprecated

from langchain_community.embeddings import HuggingFaceEmbeddings


_embedding_instance = None


def get_embeddings():
    """
    获取全局唯一 Embedding 实例
    """
    global _embedding_instance

    if _embedding_instance is None:
        _embedding_instance = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},  # 余弦相似度 / L2 统一
        )

    return _embedding_instance
