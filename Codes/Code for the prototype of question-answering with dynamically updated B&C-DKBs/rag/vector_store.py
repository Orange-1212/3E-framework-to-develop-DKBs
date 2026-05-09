# rag/vector_store.py
import faiss
import json
from typing import List, Dict
from pathlib import Path

from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.docstore.in_memory import InMemoryDocstore
from rag.embeddings import get_embeddings

# =========================
# 路径配置
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

HOT_DIR = DATA_DIR / "hot_db"
COLD_DIR = DATA_DIR / "cold_db"

# 用于保存热/冷文件列表
HOT_FILES_JSON = DATA_DIR / "hot_files.json"
COLD_FILES_JSON = DATA_DIR / "cold_files.json"


HOT_DIR.mkdir(parents=True, exist_ok=True)
COLD_DIR.mkdir(parents=True, exist_ok=True)

embeddings = get_embeddings()

# =========================
# 内部状态（文件级）
# =========================
_hot_files: List[str] = []
_cold_files: List[str] = []

if HOT_FILES_JSON.exists():
    with open(HOT_FILES_JSON, "r", encoding="utf-8") as f:
        _hot_files = json.load(f)

if COLD_FILES_JSON.exists():
    with open(COLD_FILES_JSON, "r", encoding="utf-8") as f:
        _cold_files = json.load(f)


# =========================
# 安全创建 / 加载 FAISS
# =========================
def _create_empty_db() -> FAISS:
    dim = len(embeddings.embed_query("test"))
    index = faiss.IndexFlatL2(dim)
    return FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore({}),  # 这里必须用支持 add_documents 的 docstore
        index_to_docstore_id={},
    )


def _reset_doc_ids(docs: List[Document]) -> List[Document]:
    """
    重建 FAISS 前，必须重置 Document.id
    否则会触发 Duplicate ids found
    """
    new_docs = []
    for d in docs:
        nd = Document(
            page_content=d.page_content,
            metadata=dict(d.metadata)
        )
        new_docs.append(nd)
    return new_docs


def _load_or_create(path: Path) -> FAISS:
    """
        如果目录下已有 index.faiss，就加载 FAISS 库，否则创建空库
        """
    # ✅ 确保目录一定存在（防止 Windows + FAISS 写失败）
    path.mkdir(parents=True, exist_ok=True)
    index_file = path / "index.faiss"
    if index_file.exists():
        try:
            db = FAISS.load_local(
                str(path),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            return db
        except Exception as e:
            print(f"[vector_store] 加载 FAISS 库失败，将重新创建空库: {e}")

    # 创建空库并保存
    db = _create_empty_db()
    db.save_local(str(path))
    return db

# =========================
# 全局数据库实例
# =========================
_hot_db: FAISS = _load_or_create(HOT_DIR)
_cold_db: FAISS = _load_or_create(COLD_DIR)


# =========================
# 对外：DB Getter
# =========================
def get_hot_db() -> FAISS:
    return _hot_db


def get_cold_db() -> FAISS:
    return _cold_db


# =========================
# 对外：文件列表 Getter / Setter
# =========================
def set_hot_files(files: List[str]):
    global _hot_files
    _hot_files = files
    # 保存到 JSON
    with open(HOT_FILES_JSON, "w", encoding="utf-8") as f:
        import json
        json.dump(_hot_files, f, ensure_ascii=False, indent=2)


def set_cold_files(files: List[str]):
    global _cold_files
    _cold_files = files
    with open(COLD_FILES_JSON, "w", encoding="utf-8") as f:
        import json
        json.dump(_cold_files, f, ensure_ascii=False, indent=2)


def get_hot_files() -> List[str]:
    return _hot_files


def get_cold_files() -> List[str]:
    return _cold_files



# =========================
# 写入文档（文件级控制）
# =========================
def add_documents_to_hot(docs: List[Document]):
    if docs:
        _hot_db.add_documents(docs)
        _hot_db.save_local(str(HOT_DIR))


def add_documents_to_cold(docs: List[Document]):
    if docs:
        _cold_db.add_documents(docs)
        _cold_db.save_local(str(COLD_DIR))


# =========================
# 重建数据库（供冷热切换）
# =========================
def rebuild_hot_db(docs: List[Document]):
    global _hot_db
    _hot_db = _create_empty_db()
    safe_docs = _reset_doc_ids(docs)
    add_documents_to_hot(safe_docs)


def rebuild_cold_db(docs: List[Document]):
    global _cold_db
    _cold_db = _create_empty_db()

    safe_docs = _reset_doc_ids(docs)

    add_documents_to_cold(safe_docs)


def get_all_documents() -> List[Document]:
    """
    只从 B-DKB 取全量 Document（唯一事实源）
    """
    docs: List[Document] = []

    # FAISS 内部文档存储

    if hasattr(_cold_db, "docstore"):
        docs.extend(_cold_db.docstore._dict.values())

    return list(docs)


# =========================
# 统一相似度检索接口（重要）
# =========================
def similarity_search(
    query: str,
    k: int,
    score_threshold: float,
) -> List[Document]:

    # 1️⃣ Search C-DKB
    core_hits = _hot_db.similarity_search_with_score(query, k=k)
    core_valid = [(doc, score) for doc, score in core_hits if score <= score_threshold]

    # ✅ 必须满足 k 个，才使用 C-DKB
    if len(core_valid) >= k:
        core_valid_sorted = sorted(core_valid, key=lambda x: x[1])
        return [doc for doc, _ in core_valid_sorted[:k]]

    # 2️⃣ 否则，Fallback：Search B-DKB（全量）
    background_hits = _cold_db.similarity_search_with_score(query, k=k)
    background_sorted = sorted(background_hits, key=lambda x: x[1])

    return [doc for doc, _ in background_sorted[:k]]


def delete_files(filenames: List[str]):
    """
    从 B-DKB（全量）和 C-DKB（核心）中彻底删除文件
    """
    global _cold_db

    def _remaining_docs(db: FAISS):
        if not hasattr(db, "docstore"):
            return []
        return [
            d for d in db.docstore._dict.values()
            if d.metadata.get("source") not in filenames
        ]

    # ✅ 只重建 B-DKB
    cold_docs = _remaining_docs(_cold_db)
    rebuild_cold_db(cold_docs)

    # 更新文件列表
    set_hot_files([f for f in _hot_files if f not in filenames])
    set_cold_files([f for f in _cold_files if f not in filenames])



from copy import deepcopy
from langchain.docstore.document import Document



