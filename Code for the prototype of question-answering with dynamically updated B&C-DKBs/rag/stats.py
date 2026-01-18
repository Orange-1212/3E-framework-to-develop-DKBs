# rag/stats.py
import json
from pathlib import Path
from typing import Dict, List

# =========================
# 路径
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STATS_FILE = DATA_DIR / "file_stats.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 数据结构
# =========================
# {
#   "filename.docx": {
#       "retrieval": 3,
#       "up": 2,
#       "down": 1,
#       "score": 4
#   }
# }
# =========================
def init_file_score(filename: str):
    """
    初始化上传文件的分数（score=0），如果已存在则不修改
    """
    stats = _load_stats()
    if filename not in stats:
        stats[filename] = {
            "retrieval": 0,
            "up": 0,
            "down": 0,
            "score": 0,
        }
        _save_stats(stats)


def _load_stats() -> Dict[str, Dict[str, int]]:
    if not STATS_FILE.exists():
        return {}
    with open(STATS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_stats(stats: Dict[str, Dict[str, int]]):
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def _ensure_file(stats: Dict, filename: str):
    if filename not in stats:
        stats[filename] = {
            "retrieval": 0,
            "up": 0,
            "down": 0,
            "score": 0,
        }


# =========================
# 对外接口（你在 app.py / rag_chain.py 用的）
# =========================
def record_retrieval(filename: str):
    """
    检索到一次：+1 分
    """
    stats = _load_stats()
    _ensure_file(stats, filename)

    stats[filename]["retrieval"] += 1
    stats[filename]["score"] += 1

    _save_stats(stats)


def vote_up(filename: str):
    """
    点赞：+1 分
    """
    stats = _load_stats()
    _ensure_file(stats, filename)

    stats[filename]["up"] += 1
    stats[filename]["score"] += 1

    _save_stats(stats)


def vote_down(filename: str):
    """
    点踩：-1 分
    """
    stats = _load_stats()
    _ensure_file(stats, filename)

    stats[filename]["down"] += 1
    stats[filename]["score"] -= 1

    _save_stats(stats)


# =========================
# 排行榜 & 冷热划分基础
# =========================
def get_all_file_scores() -> Dict[str, int]:
    """
    返回 {filename: score}
    """
    stats = _load_stats()
    return {k: v["score"] for k, v in stats.items()}


def get_sorted_files() -> List[str]:
    """
    按 score 从高到低排序
    """
    stats = _load_stats()
    sorted_items = sorted(
        stats.items(),
        key=lambda x: x[1]["score"],
        reverse=True,
    )
    return [filename for filename, _ in sorted_items]


def get_top_n_files(n: int) -> List[str]:
    """
    Top N 文件（热库候选）
    """
    return get_sorted_files()[:n]

def delete_files_stats(filenames: List[str]):
    """
    删除指定文件的统计信息
    """
    stats = _load_stats()
    changed = False

    for filename in filenames:
        if filename in stats:
            del stats[filename]
            changed = True

    if changed:
        _save_stats(stats)
