"""
旅行攻略知识库检索器（轻量 RAG）。

架构：
- 知识源：knowledge/ 目录下的城市攻略 Markdown（按 `## 章节` 切块）；
- 词法通道：jieba 搜索引擎分词 + rank_bm25（BM25Okapi），离线可用零成本；
- 向量通道（可选）：配置 EMBEDDING_MODEL 后经 OPENAI 兼容接口向量化，余弦相似度；
- 混合排序：两路分数各自 min-max 归一化后加权融合（0.5/0.5），向量不可用时自动退化为纯 BM25。

设计取舍：不引入向量数据库/重依赖，几百个 chunk 规模下内存余弦+BM25 足够，
且保证无 API Key 时知识库依然完整可用（与项目整体降级策略一致）。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import jieba
from loguru import logger
from rank_bm25 import BM25Okapi

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"

# 混合检索权重
BM25_WEIGHT = 0.5
VECTOR_WEIGHT = 0.5


@dataclass
class Chunk:
    """知识块：一个城市攻略的一个章节。"""

    city: str          # 所属城市（文件名）
    section: str       # 章节标题
    text: str          # 正文
    source: str        # 来源文件名（引用标注用）

    @property
    def citation(self) -> str:
        return f"{self.source}·{self.section}"


class TravelKnowledgeRetriever:
    """攻略知识库：启动时构建索引，retrieve() 返回最相关知识块。"""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._tokenized: list[list[str]] = []
        self._bm25 = None
        # 向量通道状态：None=未启用；False=启用失败已降级；list=正常
        self._vectors: Any = None
        self._build()

    # ------------------------------------------------------------
    # 索引构建
    # ------------------------------------------------------------
    def _build(self) -> None:
        self._chunks = _load_chunks(KNOWLEDGE_DIR)
        if not self._chunks:
            logger.warning("RAG 知识库为空: {}", KNOWLEDGE_DIR)
            return

        self._tokenized = [_tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(self._tokenized)
        logger.info("RAG 索引构建完成: {} 个城市, {} 个知识块", len({c.city for c in self._chunks}), len(self._chunks))

        # 可选：向量索引（EMBEDDING_MODEL 配置后才启用）
        from app.config import get_settings
        if get_settings().embedding_model:
            try:
                self._vectors = self._embed_chunks()
                logger.info("RAG 向量索引就绪 (model={})", get_settings().embedding_model)
            except Exception as e:
                self._vectors = False
                logger.warning("Embedding 初始化失败，退化为纯 BM25 检索: {}", e)

    def _embed_chunks(self) -> list[list[float]]:
        """对所有 chunk 批量向量化（OpenAI 兼容 embeddings 接口）。"""
        from langchain_openai import OpenAIEmbeddings

        from app.config import get_settings

        settings = get_settings()
        embedder = OpenAIEmbeddings(
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base,
            model=settings.embedding_model,
        )
        return embedder.embed_documents([c.text for c in self._chunks])

    # ------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------
    def retrieve(self, query: str, city: str = "", k: int = 4) -> list[dict[str, Any]]:
        """
        检索最相关的 k 个知识块。

        Args:
            query: 查询文本（如"玉龙雪山 高反 预约"）
            city: 偏好城市（非空时优先该城市的结果，但不硬过滤——跨城经验也可能相关）
            k: 返回数量

        Returns:
            [{"city", "section", "text", "citation", "score"}, ...]
        """
        if not self._bm25 or not self._chunks:
            return []

        q_tokens = _tokenize(query)

        # ---- 词法通道 ----
        bm25_scores = list(self._bm25.get_scores(q_tokens))

        # ---- 向量通道（可选）----
        vector_scores: list[float] | None = None
        if isinstance(self._vectors, list):
            try:
                q_vec = self._query_vector(query)
                vector_scores = [_cosine(q_vec, v) for v in self._vectors]
            except Exception as e:
                logger.warning("向量查询失败，本次退化为纯 BM25: {}", e)
                vector_scores = None

        # ---- 融合排序 ----
        n_bm25 = _minmax(bm25_scores)
        if vector_scores is not None and len(vector_scores) == len(self._chunks):
            n_vec = _minmax(vector_scores)
            fused = [BM25_WEIGHT * a + VECTOR_WEIGHT * b for a, b in zip(n_bm25, n_vec)]
        else:
            fused = n_bm25

        ranked = sorted(range(len(fused)), key=lambda i: fused[i], reverse=True)

        # 城市偏好：目标城市的块加权提升（而非硬过滤）
        results: list[dict[str, Any]] = []
        seen_sections: set = set()
        for idx in ranked:
            if bm25_scores[idx] <= 0:
                continue  # 与查询零词法重叠的块直接排除：
                # 未收录城市的查询不应捞到跨城噪声（如南充的行程不该出现三亚海鲜攻略）
            c = self._chunks[idx]
            boost = 1.15 if (city and c.city == city) else 1.0
            score = fused[idx] * boost
            key = (c.city, c.section)
            if key in seen_sections:
                continue
            seen_sections.add(key)
            results.append({
                "city": c.city,
                "section": c.section,
                "text": c.text,
                "citation": c.citation,
                "score": round(score, 4),
            })
            if len(results) >= k:
                break
        return results

    def _query_vector(self, query: str) -> list[float]:
        from langchain_openai import OpenAIEmbeddings

        from app.config import get_settings

        settings = get_settings()
        embedder = OpenAIEmbeddings(
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base,
            model=settings.embedding_model,
        )
        return embedder.embed_query(query)

    @property
    def status(self) -> dict[str, Any]:
        """索引状态（可观测性）。"""
        mode = "unavailable"
        if self._bm25:
            mode = "hybrid" if isinstance(self._vectors, list) else "bm25"
        return {
            "chunks": len(self._chunks),
            "cities": sorted({c.city for c in self._chunks}),
            "mode": mode,
        }


# ============================================================
# 模块级工具函数
# ============================================================
def _load_chunks(kb_dir: Path) -> list[Chunk]:
    """加载 knowledge/ 下所有 .md，按 `## 章节` 切块。"""
    chunks: list[Chunk] = []
    if not kb_dir.exists():
        return chunks

    for path in sorted(kb_dir.glob("*.md")):
        city = path.stem
        text = path.read_text(encoding="utf-8")
        # 按 ## 标题切分；标题前的引言部分归入"概述"
        parts = re.split(r"\n(?=## )", text.strip())
        for part in parts:
            part = part.strip()
            if not part:
                continue
            m = re.match(r"^## (.+)$", part.splitlines()[0])
            section = m.group(1).strip() if m else "概述"
            body = part[m.end():].strip() if m else part
            if len(body) < 20:  # 过短章节无检索价值
                continue
            chunks.append(Chunk(city=city, section=section, text=f"{section}\n{body}", source=path.name))
    return chunks


_STOPWORDS = {"的", "了", "和", "是", "在", "有", "个", "去", "要", "会", "能", "很",
              "怎么", "如何", "什么", "吗", "呢", "啊", "我", "你", "他", "们",
              "推荐", "介绍", "一下", "请问", "想", "玩", "旅游", "旅行"}


def _tokenize(text: str) -> list[str]:
    """jieba 搜索引擎分词 + 停用词/单字过滤。"""
    tokens = jieba.cut_for_search(text.lower())
    return [
        t for t in tokens
        if t.strip() and t not in _STOPWORDS and (len(t) > 1 or t.isascii())
    ]


def _minmax(scores: list[float]) -> list[float]:
    """min-max 归一化到 [0,1]；全等值时返回均匀 0.5。"""
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [0.5] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@lru_cache(maxsize=1)
def get_retriever() -> TravelKnowledgeRetriever:
    """获取全局知识库单例（进程内懒加载一次）。"""
    return TravelKnowledgeRetriever()
