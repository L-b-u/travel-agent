# Travel Agent — 基于 LangChain + LangGraph 的多 Agent 协作旅行规划系统

> 🎓 应届生求职项目 | 简历 STAR 法则：S(场景) → T(目标) → A(行动) → R(结果)

## 项目简介

采用 **LangChain 工具生态 + LangGraph 状态图编排** 的混合架构构建多 Agent 协作旅行规划系统。用户输入自然语言旅行需求（目的地、预算、天数、兴趣偏好），系统通过 **7 个专业化 Agent** 的流水线协作，自动完成偏好收集、POI 搜索、路线规划、天气查询、预算估算、行程合成和安全审查，最终输出一份带预算拆分的 Markdown 旅行计划。

**架构选型理由**：用 LangChain 的 `ChatOpenAI` + `@tool` 装饰器接入 LLM 与外部工具，享受生态成熟度；用 LangGraph 的 `StateGraph` 编排 7 节点流水线，实现 Fan-out/Fan-in 并行与安全审查拦截。这是 LangChain 官方推荐的现代多 Agent 写法。

## 📋 STAR 法则

| STAR | 内容 |
|------|------|
| **S**ituation | 自由行用户规划旅行时，需手动在 5-6 个平台间切换，单次规划耗时 3-4 小时 |
| **T**ask | 构建多 Agent 协作旅行规划系统，将规划时间压缩至 5 分钟以内 |
| **A**ction | 采用 LangChain + LangGraph 混合架构，用 `@tool` 装饰 4 个外部 API 工具、`ChatOpenAI` 接入 LLM，用 StateGraph 设计 7 节点流水线，实现安全审查三层拦截机制（敏感凭证/代操作/越界执行），编写 20 条 Eval Case |
| **R**esult | 端到端耗时 ~30s（LLM 在线）/ ~5s（降级模板），Eval 通过率 95%（19/20），安全边界用例 4/4 全拦截，7 节点全链路降级兜底（LLM 不可用仍 80% 通过） |

## 🏗️ 架构设计

```text
                        ┌──────────────────────┐
                        │   PreferenceCollector │  ← 偏好收集 Agent
                        └──────────┬───────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │     POISearcher       │  ← 高德 POI 搜索
                        └──────────┬───────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │    RoutePlanner       │  ← 高德路线规划
                        └──────┬───────┬───────┘
                               │       │
                    ┌──────────┘       └──────────┐  ← Fan-out 并行
                    ▼                              ▼
        ┌──────────────────┐          ┌──────────────────┐
        │  WeatherChecker   │          │ BudgetEstimator  │
        │  Open-Meteo API   │          │  规则+区间估算    │
        └────────┬─────────┘          └────────┬─────────┘
                 │                              │
                 └──────────────┬───────────────┘  ← Fan-in 汇聚
                                ▼
                    ┌──────────────────────┐
                    │ ItinerarySynthesizer  │  ← LLM 生成 Markdown
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   SafetyReviewer      │  ← 安全审查拦截
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Final Output        │
                    └──────────────────────┘
```

## 🔧 技术栈

| 类别 | 技术 |
|------|------|
| LLM 与工具 | **LangChain**（`ChatOpenAI` + `@tool` 装饰器 + `BaseTool` 抽象） |
| Agent 编排 | **LangGraph**（StateGraph + Fan-out/Fan-in） |
| Web 框架 | FastAPI |
| LLM 容错 | 失败重试 + 自动降级（指数退避，包裹 LangChain `ainvoke`） |
| 外部 API | 高德地图（POI 搜索 + 路线规划）、Open-Meteo（天气） |
| 数据校验 | Pydantic v2 |
| 日志 | loguru |
| 异步 HTTP | httpx |

## 📁 目录结构

```
app/
├── main.py                              # FastAPI 入口
├── config.py                            # 配置（含 Travel 相关）
├── api/routes/
│   ├── travel.py                        # 旅行规划 API（同步）
│   └── eval.py                          # 行程评估 API
├── core/
│   ├── travel/                          # 🆕 Travel Agent 核心
│   │   ├── graph.py                     # LangGraph 状态图定义
│   │   ├── state.py                     # 共享状态 Schema
│   │   ├── agents/                      # 7 个 Agent 节点
│   │   │   ├── preference_collector.py  # 偏好收集
│   │   │   ├── poi_searcher.py          # POI 搜索
│   │   │   ├── route_planner.py         # 路线规划
│   │   │   ├── weather_checker.py       # 天气查询
│   │   │   ├── budget_estimator.py      # 预算估算
│   │   │   ├── itinerary_synthesizer.py # 行程合成
│   │   │   └── safety_reviewer.py       # 安全审查
│   │   ├── tools/                       # 4 个外部 API 工具
│   │   │   ├── search_places.py         # POI 搜索
│   │   │   ├── estimate_route.py        # 路线估算
│   │   │   ├── get_weather.py           # 天气查询
│   │   │   ├── estimate_budget.py       # 预算估算
│   │   │   └── _poi_data.py             # 内置景点数据库
│   │   ├── output.py                    # 行程文件保存/读取
│   │   └── eval/                        # 评估体系
│   │       ├── cases.json               # 20 条 Eval Case
│   │       └── evaluator.py             # 评估运行器 + 独立行程质量评估
├── infrastructure/
│   └── llm/
│       └── model_router.py              # LLM 路由（失败重试 + 自动降级）
└── models/
    └── travel_schemas.py                # Travel API 模型（Pydantic）
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

编辑 `.env` 文件，至少填写：

```env
OPENAI_API_KEY=sk-xxx
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

### 3. 启动服务

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. 调用 API

**调用示例：**

```bash
curl -X POST http://127.0.0.1:8000/api/v1/travel/plan \
  -H "Content-Type: application/json" \
  -d '{"user_input": "我想去杭州玩2天，预算1000元，喜欢博物馆和美食"}'
```

### 5. 运行 Eval 评估

```bash
python -c "
import asyncio
from app.core.travel.eval import run_eval
asyncio.run(run_eval())
"
```

## 📊 Eval 评估体系

| 类型 | 用例数 | 评估重点 |
|:---|:---:|:---|
| 常规规划 | 4 | 目的地、天数、行程合理性 |
| 预算约束 | 4 | 预算是否满足、住宿等级匹配 |
| 偏好约束 | 4 | 兴趣是否纳入行程 |
| 变化处理 | 4 | 天气备选方案、异常处理 |
| 安全边界 | 4 | 付款/信息泄露拦截 |

**评估指标：**
- `constraint_satisfaction`：约束满足率
- `route_reasonableness`：路线合理性
- `source_grounding`：信息来源标注
- `uncertainty_disclosure`：不确定性说明
- `safety_compliance`：安全合规率

## 🎓 简历表达

> **Travel Agent — 基于 LangChain + LangGraph 的多 Agent 协作旅行规划系统**
>
> - 采用 **LangChain + LangGraph 混合架构**：用 LangChain `@tool` 装饰器封装 4 个外部 API（高德 POI/高德路线/Open-Meteo/预算估算）为 `BaseTool`，用 `ChatOpenAI` 统一 LLM 调用接口
> - 使用 LangGraph StateGraph 构建 7 节点 Agent 流水线，实现偏好收集→POI搜索→路线规划→天气/预算并行分析→行程合成→安全审查的自动化协作
> - 实现 Fan-out/Fan-in 并行模式（天气+预算同时查询）；安全审查节点三层拦截敏感凭证/代操作/越界执行，标记需人工确认且拒绝自动执行
> - 自实现 LLM 失败重试机制（指数退避）包裹 LangChain `ainvoke`，LLM 不可用时自动降级到规则兜底；通过 loguru 结构化日志记录各节点耗时、LLM Token 消耗与 API 调用来源（amap/降级/异常）
> - 设计覆盖 5 类场景的 20 条 Eval Case，评估约束满足率、路线合理性、安全合规率等指标
> - **技术栈**: Python, LangChain, LangGraph, FastAPI, Pydantic, OpenAI API, httpx, 高德地图, Open-Meteo

## 📝 外部 API 说明

| API | 用途 | 是否需要 Key | 文档 |
|:---|:---|:---:|:---|
| 高德地图 | POI 搜索 + 路线规划 | 免费 Key | https://lbs.amap.com/ |
| Open-Meteo | 天气查询 | 无需 | https://open-meteo.com |

**注意**：POI 搜索与路线规划**默认走高德 API**，支持全国所有城市（含雅安等地级市）。仅当无高德 API Key 或调用失败时，才降级到内置景点数据库——该库收录 36 个热门旅游目的地（杭州/北京/成都/拉萨/香格里拉等）、360+ 知名景点，作为离线兜底，**并非**全量城市覆盖。

## 许可证

MIT