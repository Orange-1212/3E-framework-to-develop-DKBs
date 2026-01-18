# app.py
import gradio as gr
from typing import List
from pathlib import Path

from rag.rag_chain import answer_question_deepseek
from rag.rag_chain import get_last_used_files
from rag.stats import (
    vote_up,
    vote_down,
    get_top_n_files,
)
from rag.vector_store import (
    get_hot_files,
    get_cold_files,
)

from rag.vector_store import get_cold_db
from rag.hot_cold_manager import (
    rebuild,
    start_auto_update,
    stop_auto_update,
)

from rag.loader import load_and_split_docs
from rag.vector_store import (
    add_documents_to_hot,
    add_documents_to_cold,
    delete_files,
)

from rag.stats import delete_files_stats

from rag.vector_store import set_cold_files

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 全局对话态
# =========================
_last_used_files: List[str] = []

# =========================
# 全局冷热参数（唯一事实源）
# =========================
CURRENT_TOP_N = 10

# =========================
# RAG 对话主函数
# =========================
def chat(question: str, k: int, score_threshold: float, api_key: str):
    global _last_used_files
    _last_used_files = []

    answer, knowledge = answer_question_deepseek(
        question=question,
        k=k,
        score_threshold=score_threshold,
        api_key=api_key,
    )

    # 返回拼接答案 + 检索知识
    return f"--- Answer ---\n\n {answer}\n\n--- Retrieved Knowledge ---\n\n{knowledge}"


# =========================
# 点赞 / 点踩
# =========================
def like_last():
    files = get_last_used_files()
    if not files:
        return "⚠️ No files available to like in this round"

    for f in files:
        vote_up(f)
    return "👍 Like recorded"


def dislike_last():
    files = get_last_used_files()
    if not files:
        return "⚠️ No files available to dislike in this round"

    for f in files:
        vote_down(f)
    return "👎 Dislike recorded"


def rebuild_with_current_top_n():
    rebuild(CURRENT_TOP_N)

# =========================
# 信息面板
# =========================
def show_hot_files():
    files = get_hot_files()
    if not files:
        return "C-DKB is currently empty"
    return "\n".join(files)


def show_cold_files():
    files = get_cold_files()
    if not files:
        return "B-DKB is currently empty"
    return "\n".join(files)


def show_top_files(n: int):
    files = get_top_n_files(n)
    if not files:
        return "No statistics available"
    return "\n".join(files)

def manual_rebuild(top_n: int):
    global CURRENT_TOP_N
    CURRENT_TOP_N = int(top_n)

    rebuild_with_current_top_n()
    return (
        f"✅ Knowledge bases rebuilt manually "
        f"(Top {CURRENT_TOP_N}). Click refresh to view the latest status."
    )


def start_auto(interval: int, top_n: int):
    global CURRENT_TOP_N
    CURRENT_TOP_N = int(top_n)

    start_auto_update(interval, CURRENT_TOP_N)

    # 刷新 UI
    hot_text = show_hot_files()
    cold_text = show_cold_files()

    return (
        f"▶️ Automatic rebuild started: every {interval}s, Top {CURRENT_TOP_N}. Click refresh to view the latest status.",
        show_hot_files(),
        show_cold_files(),
    )



def stop_auto():
    stop_auto_update()
    hot_text = show_hot_files()
    cold_text = show_cold_files()
    return "⏹ Automatic rebuild stopped", hot_text, cold_text

def safe_rebuild_after_delete():
    """
    删除文件后的安全重建：
    - 基于当前 B-DKB 文件数
    - 自动修正 top_n
    """
    from rag.vector_store import get_cold_files

    global CURRENT_TOP_N

    total_files = len(get_cold_files())
    effective_top_n = min(CURRENT_TOP_N, total_files)

    rebuild(effective_top_n)


import shutil
from pathlib import Path

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def upload_files(files):
    if not files:
        return "⚠️ No files selected", show_hot_files(), show_cold_files()

    saved_files = []

    for f in files:
        # 保存到本地文件夹，使用原始文件名
        dst = UPLOAD_DIR / Path(f.name).name
        # 用 shutil.copyfile 从临时文件拷贝内容
        shutil.copyfile(f.name, dst)
        saved_files.append(dst)

        from rag.stats import init_file_score
        init_file_score(dst.name)

    # 加载并切分
    docs = load_and_split_docs(saved_files)

    # 默认进冷库
    add_documents_to_cold(docs)

    # 自动触发一次冷热重建
    rebuild_with_current_top_n()
    upload_text = "✅ Files uploaded successfully:：\n" + "\n".join(p.name for p in saved_files)

    # 用 docstore 反推文件列表，而不是 set 拼
    set_cold_files(
        sorted(
            {d.metadata["source"] for d in get_cold_db().docstore._dict.values()}
        )
    )

    # 🔹 返回上传状态 + 热库 + 冷库
    return upload_text, show_hot_files(), show_cold_files()







def search_files(keyword: str):
    hot_files = set(get_hot_files())      # C-DKB
    all_files = set(get_cold_files())     # B-DKB（全量）

    results = []

    for f in sorted(all_files):
        if keyword and keyword not in f:
            continue

        tags = ["B-DKB"]
        if f in hot_files:
            tags.append("C-DKB")

        results.append(f"{f}  [{', '.join(tags)}]")

    return "\n".join(results) if results else "No matching files found"



def delete_file(filename: str):
    if not filename:
        return "⚠️ Please input a filename", show_hot_files(), show_cold_files()

    all_files = set(get_hot_files()) | set(get_cold_files())

    if filename not in all_files:
        return f"⚠️ File '{filename}' does not exist", show_hot_files(), show_cold_files()

    delete_files([filename])
    delete_files_stats([filename])

    # ✅ 安全重建
    safe_rebuild_after_delete()

    return (
        f"🗑 File deleted and C-DKB rebuilt: {filename}",
        show_hot_files(),
        show_cold_files(),
    )





# =========================
# Gradio UI
# =========================
with gr.Blocks(title="RAG Knowledge Base System") as demo:
    gr.Markdown(
        "<h2 style='text-align: center;'>Question-answering with dynamically updated broad-concise domain knowledge bases</h2>"
    )

    with gr.Row():
        # =====================
        # 左侧：控制面板
        # =====================

        with gr.Column(scale=1):
            # 1️⃣ 知识库状态（最上面）
            with gr.Accordion("📚 Knowledge Base Status", open=False):
                with gr.Accordion("C-DKB", open=False):
                    hot_box = gr.Textbox(label="C-DKB Files", lines=8)
                    refresh_hot = gr.Button("Refresh C-DKB")
                    refresh_hot.click(show_hot_files, outputs=hot_box)

                with gr.Accordion("B-DKB", open=False):
                    cold_box = gr.Textbox(label="B-DKB Files", lines=8)
                    refresh_cold = gr.Button("Refresh B-DKB")
                    refresh_cold.click(show_cold_files, outputs=cold_box)

            # 2️⃣ 知识库管理（上传/搜索/删除）
            with gr.Accordion("🗂 Knowledge Base Management", open=False):
                # 🔹 上传
                with gr.Accordion("📤 Upload Files", open=False):
                    upload = gr.File(
                        file_types=[".docx"],
                        file_count="multiple",
                        label="Upload Word Documents",
                    )
                    upload_status = gr.Textbox(label="Upload Status", interactive=False)
                    upload.change(
                        upload_files,
                        inputs=upload,
                        outputs=[upload_status, hot_box, cold_box],
                    )

                # 🔹 搜索
                with gr.Accordion("🔍 Search Files", open=False):
                    search_box = gr.Textbox(label="Search by Filename")
                    search_result = gr.Textbox(lines=8)
                    search_box.change(search_files, inputs=search_box, outputs=search_result)

                # 🔹 删除
                with gr.Accordion("🗑 Delete File", open=False):
                    delete_box = gr.Textbox(label="Filename to Delete")
                    delete_status = gr.Textbox(interactive=False)
                    gr.Button("Delete File").click(
                        delete_file,
                        inputs=delete_box,
                        outputs=[delete_status, hot_box, cold_box],
                    )

            # 3️⃣ 冷热库管理（手动/自动重建）
            with gr.Accordion("Knowledge Base Update Management", open=False):
                # Top N 热文件设置
                rebuild_top_n = gr.Number(value=10, precision=0, label="Top-m Files for C-DKB")
                rebuild_status = gr.Textbox(label="Status", interactive=False)

                # 手动更新
                with gr.Accordion("Manual Update", open=False):
                    rebuild_btn = gr.Button("🔄 Manual Update Knowledge Base")

                    rebuild_btn.click(
                        manual_rebuild,
                        inputs=rebuild_top_n,
                        outputs=rebuild_status,
                    )

                # 自动更新
                with gr.Accordion("Automatic Update", open=False):
                    auto_interval = gr.Number(value=300, precision=0, label="Auto Rebuild Interval (seconds)")
                    start_auto_btn = gr.Button("▶️ Start Auto Rebuild")
                    stop_auto_btn = gr.Button("⏹ Stop Auto Rebuild")

                    start_auto_btn.click(
                        start_auto,
                        inputs=[auto_interval, rebuild_top_n],
                        outputs=[rebuild_status, hot_box, cold_box],
                    )
                    stop_auto_btn.click(
                        stop_auto,
                        outputs=[rebuild_status, hot_box, cold_box],
                    )

            with gr.Accordion("⚙️ Model Settings", open=True):
                model_dropdown = gr.Dropdown(choices=["DeepSeek"], value="DeepSeek", label="LLM")
                # 🔹 将 API Key 改为密码输入
                api_input = gr.Textbox(
                    label="DeepSeek API Key",
                    placeholder="Please input API Key",
                    type="password",  # <-- 关键修改
                )

            # 4️⃣ 检索参数（K/阈值）
            with gr.Accordion("⚙️ Retrieval Settings", open=True):
                k_slider = gr.Slider(
                    minimum=1,
                    maximum=20,
                    step=1,
                    value=3,
                    label="Number of Retrieved Chunks (K)",
                )
                threshold_slider = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    step=0.05,
                    value=0.8,
                    label="Similarity Threshold (Lower is stricter)",

                )
                # 5️⃣ 文件排行榜（最下面）
                with gr.Accordion("📊 File Ranking List", open=False):
                    top_n = gr.Number(value=10, precision=0, label="Top N")
                    top_box = gr.Textbox(lines=10)
                    gr.Button("Refresh Ranking List").click(
                        show_top_files,
                        inputs=top_n,
                        outputs=top_box,
                    )

        # =====================
        # 右侧：对话区
        # =====================
        with gr.Column(scale=2):
            chatbot = gr.Textbox(
                label="Answer",
                lines=18,
                interactive=False,
            )

            question_box = gr.Textbox(
                label="Enter your question",
                placeholder="Enter your question and press Enter",
            )

            gr.Markdown(
                "💡 **Feedback**:<br>"
                "If the retrieved knowledge is useful, please click 👍. If it is useless, please click 👎."
            )
            with gr.Row():
                like_btn = gr.Button("👍")
                dislike_btn = gr.Button("👎")

            # 绑定对话
            question_box.submit(
                chat,
                inputs=[question_box, k_slider, threshold_slider, api_input],
                outputs=chatbot,
            )

            like_btn.click(like_last, outputs=chatbot)
            dislike_btn.click(dislike_last, outputs=chatbot)



            rebuild_btn.click(
                manual_rebuild,
                inputs=rebuild_top_n,
                outputs=rebuild_status,
            )





# =========================
# 启动
# =========================
if __name__ == "__main__":
    demo.launch()


