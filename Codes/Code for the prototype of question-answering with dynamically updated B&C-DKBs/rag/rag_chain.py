# rag/rag_chain.py

import requests
from typing import List, Generator

from langchain.docstore.document import Document

from rag.vector_store import similarity_search
from rag.stats import record_retrieval

import os
from openai import OpenAI

_last_used_files: List[str] = []


def get_last_used_files() -> List[str]:
    """
    返回最近一次问答中命中的文件列表
    """
    return _last_used_files
# =========================
# LLM 接口预留（Q1 用 A，Q2 用 B）
# =========================
def call_deepseek(prompt: str, api_key: str, model: str = "deepseek-chat") -> str:
    """
    使用官方 SDK 调用 DeepSeek
    """
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个专业的领域助手"},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ 调用 DeepSeek API 异常: {e}"

# =========================
# 构造 Prompt
# =========================
def build_prompt(question: str, docs: List[Document]) -> str:
    context_blocks = [
        f"{i + 1}. {doc.page_content}" for i, doc in enumerate(docs)
    ]
    context = "\n\n".join(context_blocks)

    prompt = f"""
参考以下知识回答问题。

{context}

问题：{question}


"""
    return prompt.strip()


# =========================
# 对外统一入口（app.py 调用）
# =========================
def answer_question_deepseek(
    question: str,
    k: int,
    score_threshold: float,
    api_key: str,
):
    """
    RAG 主流程 + DeepSeek 接入

    返回：
    - answer: DeepSeek 返回的答案
    - knowledge_text: 检索到的知识 + 来源
    """
    # 1️⃣ 检索知识
    docs = similarity_search(question, k=k, score_threshold=score_threshold)

    if not docs:
        return "❌ 未检索到满足条件的知识内容。", ""

    # 2️⃣ 记录文件级检索分数
    used_files = set()
    for doc in docs:
        filename = doc.metadata.get("source")
        if filename and filename not in used_files:
            record_retrieval(filename)
            used_files.add(filename)

    global _last_used_files
    _last_used_files = list(used_files)

    # 3️⃣ 构造 Prompt
    prompt = build_prompt(question, docs)

    # 4️⃣ 调用 DeepSeek
    answer = call_deepseek(prompt, api_key=api_key)

    # 5️⃣ 返回答案 + 检索知识与来源
    knowledge_text = "\n\n".join([f"{doc.metadata.get('source')}: {doc.page_content}" for doc in docs])
    return answer, knowledge_text