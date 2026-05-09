# rag/hot_cold_manager.py


import threading
import time
from typing import List

from rag.stats import get_top_n_files
from rag.vector_store import (
    set_hot_files,
    set_cold_files,
    rebuild_hot_db,
    rebuild_cold_db,
    get_all_documents,
)


# =========================
# 全局状态
# =========================
_auto_thread = None
_stop_event = threading.Event()



# =========================
# 核心逻辑
# =========================
def rebuild(top_n: int):
    """
    重建 C-DKB（核心库）
    ❗不允许修改 B-DKB
    """
    from rag.stats import get_top_n_files
    from rag.vector_store import (
        get_all_documents,
        rebuild_hot_db,
        set_hot_files,
    )

    # 1️⃣ Top-N 核心文件
    hot_files = get_top_n_files(top_n)

    # 2️⃣ 从 B-DKB（全量）中选核心文档
    all_docs = get_all_documents()
    hot_docs = [
        d for d in all_docs
        if d.metadata.get("source") in hot_files
    ]

    # 3️⃣ 只重建 C-DKB
    rebuild_hot_db(hot_docs)
    set_hot_files(hot_files)

    # ❌ 不得 set_cold_files





# =========================
# 自动更新
# =========================
def _auto_loop(interval_seconds: int, top_n: int):
    while not _stop_event.is_set():
        try:
            rebuild(top_n)
        except Exception as e:
            print(f"[HotColdManager] 自动更新失败: {e}")
        # 等待下一轮
        _stop_event.wait(interval_seconds)


def start_auto_update(interval_seconds: int, top_n: int):
    global _auto_thread

    if _auto_thread and _auto_thread.is_alive():
        return

    _stop_event.clear()
    _auto_thread = threading.Thread(
        target=_auto_loop,
        args=(interval_seconds, top_n),
        daemon=True,
    )
    _auto_thread.start()

    print(f"[HotColdManager] 自动重建线程已启动，每 {interval_seconds}s 执行一次，Top {top_n}")



def stop_auto_update():
    """
    停止自动冷热更新
    """
    _stop_event.set()
