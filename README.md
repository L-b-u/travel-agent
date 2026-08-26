# Travel Agent — 基于 LangChain + LangGraph 的 LLM Agent 工程化实践

> 🎓 应届生求职项目 | 题材是载体，工程实践是核心：Workflow vs Agent 取舍、HITL 落地、双层安全审查、Eval 评估体系、全链路降级

## 项目简介

用户输入自然语言旅行需求（目的地、预算、天数、兴趣偏好），系统通过 LangGraph 状态图编排多个专业化节点，自动完成偏好提取、POI 搜索、路线估算、攻略检索、天气/预算分析、行程合成与安全审查，输出一份带预算拆分和引用来源的 Markdown 旅行计划。

**这不是固定流水线套壳**——研究阶段是一个真正的 ReAct Agent：LLM 通过 Tool Calling 自主决定调用哪些工具、调用几次（先查什么、后查什么、要不要验证两点间距离）；而天气/预算等必须保证覆盖的关键数据走确定性并行节点。两种范式的取舍依据写在 [架构设计](#️-架构设计) 一节。

## ✨ 核心亮点

| 能力 | 实现 | 为什么值得讲 |
|------|------|--------------|
| **真实 Tool Calling** | `create_react_agent` + `bind_tools`，LLM 自主调度 POI 搜索/路线估算/RAG 检索三个工具，结果经 side-channel 捕获不依赖 LLM 复述 | Agent 自主性有代码事实支撑，side-channel 是防幻觉丢数据的实战手法 |
| **HITL 人工确认闭环** | 审查命中 → `interrupt()` 中断 → 用户批准/拒绝 → `Command(resume=...)` 恢复；输入侧风险批准后可继续规划（条件边路由回主流程） | 用的是 LangGraph 区别于其他框架的核心机制，且处理了"批准后继续"与防重复拦截 |
| **Fail-fast 入口守卫** | 危险请求在消耗任何 LLM/API 资源前被拦截（实测 0.02s，对比末端拦截 ~60s） | 安全前置是生产思维；测试断言了拦截耗时 |
| **双层安全审查** | 输入侧（入口）+ 输出侧（合成后）分别检测；分级处置：确认/仅提醒/放行；正常订房请求不误伤（有回归用例锁定） | 安全不是关键词黑名单一句话，误伤修复过程本身就是好故事 |
| **RAG 攻略知识库** | 8 城市 Markdown 攻略按章节切块，jieba 分词 + BM25 词法检索，配置 Embedding 后自动升级为混合检索；作为 Agent 工具暴露 + 合成上下文确定性注入，行程尾部标注引用来源 | 轻量无重依赖（无向量库），降级策略与全局一致；source grounding 可验证 |
| **结构化输出** | 偏好收集走 `with_structured_output` 返回 Pydantic 实例，Schema 内做枚举归一化与非法值过滤 | 不用手写正则抠 JSON；LLM 输出非法时整体降级规则兜底 |
| **Eval 双层评估** | 规则检查（结构/约束/安全合规，毫秒级回归）+ LLM-as-Judge rubric 五维评分（约束满足/可执行性/信息完备/应变/表达），汇总进报告 | 规则保底线、judge 补深度，两层互补 |
| **全链路降级** | LLM 失败→规则兜底；高德失败→内置 360+ 景点库；Open-Meteo 失败→占位天气；Embedding 失败→纯 BM25。任一外部依赖挂掉系统仍可用 | 测试全部离线可跑就是这套设计的直接证明 |
| **可观测性** | loguru 结构化日志（节点耗时/Token 用量/数据来源）；一行环境变量接入 LangSmith 或 Langfuse trace | 从日志级到平台级的观测路径都打通 |

## 🏗️ 架构设计

```text
                    ┌────────────────────┐
        user_input →│    input_guard      │ 入口守卫：资金执行/敏感凭证 fail-fast
                    └─────────┬──────────┘
                 (通过)│        │(风险)
                      │        ▼
                      │   human_gate ◀── interrupt() 中断，等待人工决定
                      ▼        │(批准·继续)
            ┌────────────────────┐
            │ collect_preferences │ LLM 结构化输出（Pydantic Schema）
            └─────────┬──────────┘
                      ▼
            ┌────────────────────┐
            │      research       │ ReAct Agent（create_react_agent）
            │  search_pois ┐      │ LLM 自主决定工具调用顺序与次数
            │  estimate_route ├─▶ side-channel 捕获结构化数据
            │  search_travel_tips │ RAG 攻略检索
            └─────────┬──────────┘
              Fan-out │ （确定性节点，保证关键数据覆盖）
         ┌────────────┴────────────┐
         ▼                         ▼
   check_weather             estimate_budget
   (Open-Meteo)              （规则+城市物价指数）
         └────────────┬────────────┘
                Fan-in ▼
            ┌────────────────────┐
            │     synthesize      │ LLM 生成 Markdown 行程（支持 token 流式）
            └─────────┬──────────┘
                      ▼
            ┌────────────────────┐
            │    safety_review    │ 输出侧审查：越界表述/索要凭证
            └─────────┬──────────┘
                 (通过)│        │(风险)
                      │        ▼
                      │   human_gate ── 批准：追加确认记录交付
                      │               └─ 拒绝：取消说明（status=cancelled）
                      ▼
                     END
```

**Workflow 与 Agent 的取舍**：探索性任务（搜什么、按什么顺序看）交给 ReAct Agent 收益来自动态决策；天气/预算是必须覆盖的关键数据，走确定性并行节点更可控、可测、可降级。整张图里两种范式各司其职，而不是为了"多 Agent"名头把所有环节都包成节点壳。

## 🔧 技术栈

| 类别 | 技术 |
|------|------|
| LLM 接入 | LangChain `ChatOpenAI`（文本生成 / 结构化输出 / Tool Calling 统一路由） |
| Agent 编排 | LangGraph `StateGraph` · `interrupt()` HITL · `create_react_agent` · Fan-out/Fan-in |
| RAG | jieba 分词 + rank_bm25，可选 OpenAI 兼容 Embedding 混合检索（无向量库依赖） |
| Web 框架 | FastAPI（同步端点 / SSE 流式 / 文件上传） |
| 容错 | 指数退避重试 + 分层降级（LLM→规则，API→内置库，向量→BM25） |
| 数据校验 | Pydantic v2（API 模型 / 结构化输出 Schema / 配置） |
| 测试 | pytest + pytest-asyncio（73 例，全部离线，外部依赖工具边界打桩） |
| 质量工具 | ruff（E/F/I/UP 全绿）· GitHub Actions（lint + test + docker build） |
| 前端 Demo | Streamlit（SSE 流式渲染 + 确认交互面板） |
| 外部 API | 高德地图（POI/路线）、Open-Meteo（天气） |

## 📁 目录结构

```
app/
├── main.py                       # FastAPI 入口
├── config.py                     # pydantic-settings 配置（含观测/RAG 开关）
├── api/routes/
│   ├── travel.py                 # /plan 同步 · /plan/stream SSE · /confirm HITL
│   └── eval.py                   # 行程质量即时评估（纯规则）
├── core/travel/
│   ├── graph.py                  # LangGraph 状态图 + interrupt/resume 编排
│   ├── state.py                  # 共享状态 TypedDict
│   ├── agents/
│   │   ├── preference_collector.py   # 结构化输出偏好提取（规则兜底）
│   │   ├── research_agent.py         # ⭐ ReAct Tool Calling 研究 Agent
│   │   ├── weather_checker.py        # 天气查询（Fan-out）
│   │   ├── budget_estimator.py       # 预算估算（Fan-out）
│   │   ├── itinerary_synthesizer.py  # 行程合成（token 流式钩子）
│   │   └── safety_reviewer.py        # 双层审查 + 入口守卫（纯函数可测）
│   ├── tools/                    # 高德 POI/路线、Open-Meteo、预算规则
│   ├── rag/
│   │   ├── retriever.py          # BM25 + 可选向量混合检索
│   │   └── knowledge/*.md        # 8 城市攻略知识库（40 章节）
│   └── eval/
│       ├── cases.json            # 22 条 Eval Case（5 类场景 + 误伤回归）
│       ├── evaluator.py          # 规则评估运行器 + 独立评估接口
│       └── judge.py              # LLM-as-Judge 五维 rubric 评分
├── infrastructure/llm/model_router.py  # 统一 LLM 接入（重试/流式/结构化/Langfuse）
├── models/travel_schemas.py      # API 请求/响应模型
frontend/app.py                  # Streamlit Demo（SSE 流式 + 确认面板）
tests/                           # 73 个离线测试（conftest 统一打桩）
```

## 🚀 快速开始

### 1. 安装

```bash
pip install -e ".[dev,observability,frontend]"
```

### 2. 配置 `.env`

```env
# LLM（必填其一，不填则全流程走规则兜底，功能可用但无智能）
OPENAI_API_KEY=sk-xxx
OPENAI_API_BASE=https://api.openai.com/v1     # 兼容任意 OpenAI 格式端点
OPENAI_MODEL=gpt-4o-mini

# 高德地图（免费申请：https://lbs.amap.com/）
AMAP_API_KEY=xxx

# 可选：RAG 向量通道（留空则纯 BM25）
EMBEDDING_MODEL=text-embedding-3-small

# 可选：可观测性
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_xxx
# 或 Langfuse：
# LANGFUSE_PUBLIC_KEY=pk-lf-xxx
# LANGFUSE_SECRET_KEY=sk-lf-xxx
```

### 3. 启动后端

```bash
uvicorn app.main:app --reload --port 8000
```

### 4. 启动前端 Demo（可选）

```bash
streamlit run frontend/app.py
```

### 5. Docker

```bash
docker build -t travel-agent .
docker run -p 8000:8000 --env-file .env travel-agent
```

## 📡 API

| 端点 | 说明 |
|:---|:---|
| `POST /api/v1/travel/plan` | 同步规划；命中安全审查时返回 `status=pending_confirmation` + 待确认项 |
| `POST /api/v1/travel/confirm` | 提交人工确认决定（HITL 恢复）；批准继续 / 拒绝取消 |
| `POST /api/v1/travel/plan/stream` | SSE 流式规划：delta（增量 Markdown）/ result / confirm / error 事件 |
| `POST /api/v1/eval/itinerary` | 对任意 Markdown 行程做规则质量评估（毫秒级） |

示例：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/travel/plan \
  -H "Content-Type: application/json" \
  -d '{"user_input": "我想去成都玩2天，预算1500元，喜欢美食和大熊猫", "session_id": "demo"}'
```

## 📊 Eval 评估体系

22 条用例 × 5 类场景，规则检查与 LLM 评审并行输出：

| 类型 | 用例数 | 评估重点 |
|:---|:---:|:---|
| 常规规划 | 4 | 目的地/天数体现、POI 数据落地（非编造） |
| 预算约束 | 4 | 预算金额呈现、住宿等级匹配 |
| 偏好约束 | 4 | 兴趣纳入行程、负向约束（不要购物） |
| 变化处理 | 4 | 天气备选方案、健康提醒 |
| 安全边界 | 6 | 敏感操作拦截（4）+ **无误伤放行回归（2）** |

```bash
python run_eval.py                        # 全量（规则 + LLM judge）
python run_eval.py --type 安全边界        # 单类型
python run_eval.py --no-judge            # 只跑规则检查（快）
```

报告包含分类通过率与 judge 五维均分（约束满足 / 可执行性 / 信息完备 / 应变能力 / 表达质量）。

## 🧪 测试

```bash
pytest tests/ -q          # 无需任何 API Key，全离线，<10s
ruff check app tests
```

测试策略：外部依赖在高德/Open-Meteo/LLM 的**工具边界**统一打桩（`tests/conftest.py`），单测覆盖安全审查误伤回归、结构化输出校验、RAG 相关性、HITL 中断/恢复/fail-fast 时限、API 端点行为（含无断点恢复的 409 防御）。GitHub Actions 在 push/PR 时跑 lint + 双版本测试 + Docker 构建。

## 🎓 简历表达

> **Travel Agent — 基于 LangChain + LangGraph 的 LLM Agent 工程化实践**
>
> - 使用 LangGraph StateGraph 编排混合范式流水线：研究阶段以 create_react_agent 构建真正的 Tool Calling Agent（LLM 自主决策调用 POI 搜索/路线估算/RAG 检索），天气/预算等关键数据走确定性 Fan-out/Fan-in 并行节点，兼顾探索性与可控性
> - 落地 HITL 人工确认闭环：双层安全审查（入口守卫 fail-fast + 输出侧复查）通过 interrupt() 中断图执行，人工批准后条件边路由回主流程继续规划；危险请求拦截从 ~60s 降至 0.02s，并有误伤回归用例锁定正常订房请求不受影响
> - 实现轻量 RAG 攻略知识库（jieba+BM25，可选 Embedding 混合检索，无向量库依赖），以 Agent 工具 + 合成上下文双通道注入，行程标注引用来源实现 source grounding
> - 设计双层 Eval 体系：22 条规则用例（毫秒级回归拦截）+ LLM-as-Judge 五维 rubric 评分；偏好收集使用 with_structured_output 结构化输出并内建枚举归一化
> - 全链路降级设计（LLM→规则兜底、地图 API→内置数据库、向量→BM25），73 个离线单元测试 <10s 跑完，GitHub Actions 自动化 lint/test/docker 构建
>
> **技术栈**: Python, LangChain, LangGraph, FastAPI(SSE), Pydantic v2, RAG(BM25), pytest, Docker, 高德地图, Open-Meteo

## 许可证

MIT
