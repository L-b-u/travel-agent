"""
Travel Agent 前端 Demo（Streamlit）。

运行：
    streamlit run frontend/app.py

功能：
- 自然语言输入旅行需求 → SSE 流式渲染行程 Markdown（打字机效果）
- 安全审查面板：需人工确认时展示待确认项，批准/拒绝走 /confirm 端点
- 展示研究模式（react/deterministic）、工具调用轨迹与 RAG 引用来源
"""

from __future__ import annotations

import json
import uuid

import httpx
import streamlit as st

API_BASE = "http://127.0.0.1:8000/api/v1"

# trust_env=False：绕过系统代理。httpx 默认读取 HTTP_PROXY 等环境变量，
# 会把发往 127.0.0.1 的请求也交给代理，导致 502 / 空响应（本地 Demo 常见坑）
client = httpx.Client(trust_env=False, timeout=300)

st.set_page_config(page_title="Travel Agent", page_icon="🧳", layout="wide")
st.title("🧳 Travel Agent")
st.caption("LangGraph 多 Agent 旅行规划 · ReAct 工具调用 · RAG 攻略检索 · HITL 安全确认")

# ---------------------------------------------------------------------------
# 会话状态
# ---------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = f"web_{uuid.uuid4().hex[:8]}"
if "pending" not in st.session_state:
    st.session_state.pending = None  # {"items": [...], "safety_result": {...}}


def stream_plan(user_input: str, session_id: str, placeholder) -> dict | None:
    """调 SSE 接口，逐段渲染行程，返回最终 result 数据（confirm 时返回 None）。"""
    final = None
    text = ""
    try:
        with client.stream(
            "POST",
            f"{API_BASE}/travel/plan/stream",
            json={"user_input": user_input, "session_id": session_id},
        ) as resp:
            if resp.status_code != 200:
                body = resp.read().decode("utf-8", errors="replace")[:300]
                placeholder.error(f"后端返回 HTTP {resp.status_code}: {body}")
                return None

            got_terminal_event = False
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                try:
                    event = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                etype = event["type"]
                if etype == "delta":
                    text += event["text"]
                    placeholder.markdown(text + "▌")
                elif etype == "result":
                    final = event["data"]
                    placeholder.markdown(final["itinerary"])
                    got_terminal_event = True
                elif etype == "confirm":
                    st.session_state.pending = {
                        "items": event.get("items", []),
                        "safety_result": event.get("safety_result", {}),
                    }
                    placeholder.empty()
                    # 立即重渲染页面：确认按钮面板在脚本前段，pending 是在这里才设置的，
                    # 不主动触发 rerun 的话按钮要等下一次交互才会出现
                    st.rerun()
                elif etype == "error":
                    placeholder.error(f"规划失败: {event.get('message')}")
                    return None

            if not got_terminal_event:
                # 连接提前断开且无任何终态事件：多半是后端中途崩溃或被代理截断
                placeholder.error(
                    "连接中断且未收到完整结果。\n\n"
                    f"请确认后端已在 8000 端口运行：`uvicorn app.main:app --port 8000`\n\n"
                    f"已接收内容长度：{len(text)} 字符"
                )
    except httpx.HTTPError as e:
        placeholder.error(
            f"无法连接后端 `{API_BASE}`：{e!r}\n\n"
            "请先启动服务：`uvicorn app.main:app --port 8000`"
        )
    return final


def render_meta(result: dict) -> None:
    """展示研究元信息与引用来源。"""
    meta = result.get("research_meta", {})
    col1, col2, col3 = st.columns(3)
    col1.metric("研究模式", "ReAct 工具调用" if meta.get("mode") == "react" else "确定性流水线")
    col2.metric("工具调用次数", meta.get("tool_calls", "-"))
    tips = result.get("tips_citations", [])
    col3.metric("攻略引用", len(tips))
    if tips:
        st.info("📚 攻略来源：" + "、".join(tips))


def submit_decision(approved: bool) -> None:
    """提交人工确认决定并展示结果。"""
    st.session_state.pop("pending")
    with st.spinner("已确认，继续处理…" if approved else "正在取消…"):
        resp = client.post(
            f"{API_BASE}/travel/confirm",
            json={"session_id": st.session_state.session_id, "approved": approved},
        )
    if resp.status_code != 200:
        st.error(f"确认失败: {resp.text}")
        return
    data = resp.json()
    if data.get("status") == "cancelled":
        st.warning(data["itinerary"])
    else:
        st.markdown(data["itinerary"])
        render_meta(data)


# ---------------------------------------------------------------------------
# 主界面
# ---------------------------------------------------------------------------
user_input = st.text_area(
    "描述你的旅行需求",
    placeholder="例：我想去成都玩2天，预算1500元，喜欢美食和大熊猫",
    height=80,
)

col_btn, col_session = st.columns([1, 2])
plan_clicked = col_btn.button("开始规划 🚀", type="primary", use_container_width=True)
col_session.caption(f"会话 ID：`{st.session_state.session_id}`")

placeholder = st.empty()

# 恢复中的确认流优先展示
if st.session_state.pending:
    items = st.session_state.pending["items"]
    with st.container(border=True):
        st.error("**🔒 需要你确认**——以下请求涉及资金或账户安全，系统不会代为执行：")
        for item in items:
            st.write(f"- {item}")
        a, b = st.columns(2)
        if a.button("✅ 我知道了，继续生成行程", use_container_width=True):
            submit_decision(True)
        if b.button("❌ 取消本次规划", use_container_width=True):
            submit_decision(False)

elif plan_clicked and user_input.strip():
    with st.status("规划中…", expanded=False) as status:
        st.write("1️⃣ 提取结构化偏好（LLM 结构化输出）")
        st.write("2️⃣ 研究：搜索 POI / 估算路线 / 检索攻略（ReAct Tool Calling）")
        st.write("3️⃣ 天气 + 预算并行分析（Fan-out/Fan-in）")
        st.write("4️⃣ 合成行程 Markdown → 安全审查 → 交付")
        result = stream_plan(user_input, st.session_state.session_id, placeholder)
        if result:
            status.update(label="✅ 规划完成", state="complete", expanded=False)
            render_meta(result)
        elif st.session_state.pending:
            status.update(label="⏸️ 等待人工确认", state="complete")
        else:
            status.update(label="❌ 规划失败", state="error")
elif plan_clicked:
    st.warning("请先输入旅行需求")

with st.sidebar:
    st.header("使用说明")
    st.markdown(
        """
1. 启动后端：`uvicorn app.main:app --port 8000`
2. 输入需求开始规划；行程会**流式**生成
3. 试一试安全拦截：
   > 帮我订机票，用我的信用卡直接付款

   系统会中断等待你的决定（HITL）
4. 正常请求如"帮我订酒店"不会误伤
"""
    )
    st.divider()
    st.caption(
        "技术栈：LangChain · LangGraph (StateGraph/interrupt/create_react_agent)\n\n"
        "FastAPI SSE · BM25+向量混合检索 · pytest"
    )
