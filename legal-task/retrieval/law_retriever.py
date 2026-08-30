import os
import json
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np

# Optional dependencies (fallbacks will be used if unavailable)
try:
    import faiss  # faiss-cpu or faiss-gpu
except Exception:
    faiss = None

try:
    import jieba
except Exception:
    jieba = None

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

from mas.utils import EmbeddingFunc


@dataclass
class LawArticleRetriever:
    """
    法条检索器（优化版）：
    - 从 JSONL 文件加载法条（字段：law_article_id, law_article_content）
    - 使用 SentenceTransformer/EmbeddingFunc 对法条进行向量化
    - 构建 FAISS 向量索引（如不可用则退化为 numpy 余弦相似检索）
    - 可选：构建 BM25 词法索引并支持简单融合（rrf/zscore/minmax），默认 dense-only
    - 保持原有接口：retrieve(text, top_k) 返回 [(id, content), ...]
    """

    law_jsonl_path: str
    persist_dir: str
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # 可选：直接传入外部构造好的嵌入函数，避免重复加载模型
    embed_func: Optional[EmbeddingFunc] = None
    # 可选：当未传入 embed_func 时，用于创建 EmbeddingFunc 的默认设备与批量
    embedding_device: Optional[str] = None
    embedding_batch_size: int = 16
    use_bm25: bool = False
    bm25_top_k: int = 50
    faiss_top_k: int = 100
    fusion_alpha: float = 0.7  # 用于融合时的加权

    def __post_init__(self):
        os.makedirs(self.persist_dir, exist_ok=True)
        # 载入语料
        self._articles: List[Dict[str, Any]] = []
        self._id_to_idx: Dict[str, int] = {}
        self._load_corpus()

        # 构建向量检索索引
        # 优先使用外部传入的 embed_func，避免重复将大模型加载到 GPU0
        if self.embed_func is None:
            # 从环境中读取默认设备与批量；若未设置则使用传入的字段
            device = self.embedding_device or os.environ.get('EMBEDDING_DEVICE')
            try:
                batch_env = int(os.environ.get('EMBEDDING_BATCH_SIZE', str(self.embedding_batch_size)))
            except Exception:
                batch_env = self.embedding_batch_size
            self.embed_func = EmbeddingFunc(
                self.embedding_model,
                device=device,
                batch_size=batch_env,
            )
        self._faiss_index: Optional[Any] = None
        self._emb_matrix: Optional[np.ndarray] = None
        self._build_dense_index()

        # 构建 BM25（如启用且可用）
        self._bm25_index: Optional[Any] = None
        if self.use_bm25 and BM25Okapi is not None and jieba is not None:
            self._build_bm25_index()

    # ------------------------ Corpus & Index Builders ------------------------
    def _load_corpus(self) -> None:
        if not os.path.exists(self.law_jsonl_path):
            raise FileNotFoundError(f"Mapping file not found: {self.law_jsonl_path}")
        with open(self.law_jsonl_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "law_article_id" in obj and "law_article_content" in obj:
                        self._articles.append(obj)
                        self._id_to_idx[str(obj["law_article_id"])] = len(self._articles) - 1
                except json.JSONDecodeError:
                    # 跳过不合法行
                    continue
        if not self._articles:
            raise ValueError("No valid articles found in mapping file")

    def _build_dense_index(self) -> None:
        texts = [str(a["law_article_content"]) for a in self._articles]
        vecs = self.embed_func.embed_documents(texts)
        X = np.asarray(vecs, dtype=np.float32)
        # 归一化以使用内积近似余弦相似
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X = X / norms
        dim = X.shape[1]

        if faiss is not None:
            # 使用内积索引（归一化后等价于余弦相似）
            index = faiss.IndexFlatIP(dim)
            index.add(X)
            self._faiss_index = index
            self._emb_matrix = None
        else:
            # 退化到 numpy 检索
            self._faiss_index = None
            self._emb_matrix = X

    def _build_bm25_index(self) -> None:
        tokenized_corpus = []
        for a in self._articles:
            tokens = list(jieba.cut(str(a["law_article_content"])))
            tokenized_corpus.append(tokens)
        self._bm25_index = BM25Okapi(tokenized_corpus)

    # ------------------------ Search Methods ------------------------
    def _dense_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        q = np.asarray([self.embed_func.embed_query(query)], dtype=np.float32)
        # 归一化
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        if self._faiss_index is not None:
            # 限制检索量以避免过慢
            k = min(max(top_k, 1), min(self.faiss_top_k, len(self._articles)))
            scores, indices = self._faiss_index.search(q, k)
            res = []
            for s, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                if 0 <= idx < len(self._articles):
                    res.append((idx, float(s)))
            return res
        else:
            # numpy 余弦相似检索
            if self._emb_matrix is None or self._emb_matrix.size == 0:
                return []
            sims = (self._emb_matrix @ q[0])  # 内积等价余弦
            idxs = np.argsort(-sims)[:top_k]
            return [(int(i), float(sims[i])) for i in idxs]

    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        if self._bm25_index is None:
            return []
        tokens = list(jieba.cut(query))
        scores = self._bm25_index.get_scores(tokens)
        idxs = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in idxs]

    def _fuse(self, dense: List[Tuple[int, float]], sparse: List[Tuple[int, float]], mode: str = "rrf") -> List[int]:
        # 简单融合：支持 rrf/minmax/zscore，默认 rrf
        if not sparse:
            return [idx for idx, _ in dense]
        if not dense:
            return [idx for idx, _ in sparse]

        # 建立分数表
        d_scores: Dict[int, float] = {idx: s for idx, s in dense}
        s_scores: Dict[int, float] = {idx: s for idx, s in sparse}
        all_ids = list(set(d_scores.keys()) | set(s_scores.keys()))

        if mode == "minmax":
            def norm(scores: Dict[int, float]) -> Dict[int, float]:
                vals = np.array(list(scores.values()), dtype=np.float32)
                if vals.size == 0:
                    return {k: 0.0 for k in scores}
                mn, mx = float(vals.min()), float(vals.max())
                rng = mx - mn if mx > mn else 1.0
                return {k: (v - mn) / rng for k, v in scores.items()}
            d_n = norm(d_scores)
            s_n = norm(s_scores)
            fused = {i: self.fusion_alpha * d_n.get(i, 0.0) + (1 - self.fusion_alpha) * s_n.get(i, 0.0) for i in all_ids}
        elif mode == "zscore":
            def z(scores: Dict[int, float]) -> Dict[int, float]:
                vals = np.array(list(scores.values()), dtype=np.float32)
                if vals.size == 0:
                    return {k: 0.0 for k in scores}
                mu, sd = float(vals.mean()), float(vals.std())
                sd = sd if sd > 1e-6 else 1.0
                return {k: (v - mu) / sd for k, v in scores.items()}
            d_z = z(d_scores)
            s_z = z(s_scores)
            fused = {i: self.fusion_alpha * d_z.get(i, 0.0) + (1 - self.fusion_alpha) * s_z.get(i, 0.0) for i in all_ids}
        else:
            # Reciprocal Rank Fusion (RRF)
            # 先按分数降序取排序名次
            d_rank = {idx: r for r, (idx, _) in enumerate(sorted(dense, key=lambda x: x[1], reverse=True), start=1)}
            s_rank = {idx: r for r, (idx, _) in enumerate(sorted(sparse, key=lambda x: x[1], reverse=True), start=1)}
            fused = {i: self.fusion_alpha * (1.0 / (d_rank.get(i, len(d_rank)) + 60)) +
                        (1 - self.fusion_alpha) * (1.0 / (s_rank.get(i, len(s_rank)) + 60))
                     for i in all_ids}

        return [i for i, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)]

    # ------------------------ Public API ------------------------
    def retrieve(self, text: str, top_k: int = 5, ranking_mode: str = "dense") -> List[Tuple[int, str]]:
        # Dense 检索
        dense_res = self._dense_search(text, top_k=max(top_k, 1))
        # 可选：BM25 检索
        sparse_res: List[Tuple[int, float]] = []
        if self.use_bm25 and self._bm25_index is not None:
            sparse_res = self._bm25_search(text, top_k=min(self.bm25_top_k, max(top_k, 1)))

        # 融合/选择
        if self.use_bm25 and sparse_res:
            order = self._fuse(dense_res, sparse_res, mode=ranking_mode)
        else:
            order = [idx for idx, _ in dense_res]

        # 根据排序组装返回（id, content）
        candidates: List[Tuple[int, str]] = []
        for i in order[:top_k]:
            art = self._articles[int(i)]
            candidates.append((int(art["law_article_id"]), str(art["law_article_content"])) )
        return candidates

    def get_articles_by_ids(self, ids: List[int]) -> List[Tuple[int, str]]:
        result: List[Tuple[int, str]] = []
        if not ids:
            return result
        for id_val in ids:
            key = str(id_val)
            idx = self._id_to_idx.get(key)
            if idx is None:
                continue
            art = self._articles[int(idx)]
            result.append((int(art["law_article_id"]), str(art["law_article_content"])) )
        return result

    @staticmethod
    def _extract_crimes(content: str) -> List[str]:
        import re
        crimes: List[str] = []
        # 提取中文方括号中的片段，如【诈骗罪】、【盗窃罪】等
        for m in re.findall(r"【([^】]+)】", content or ""):
            name = m.strip()
            # 过滤明显非罪名的描述（如“处罚规定”），保留包含“罪”的片段
            if "罪" in name and "规定" not in name:
                # 保留完整罪名，不对方括号内的“、/，”进行拆分
                crimes.append(name)
        # 去重并保持顺序
        seen = set()
        ordered = []
        for c in crimes:
            if c not in seen:
                ordered.append(c)
                seen.add(c)
        return ordered

    @staticmethod
    def format_context(candidates: List[Tuple[int, str]], max_chars: int = 400) -> str:
        lines = []
        for law_id, content in candidates:
            snippet = (content or "").strip()
            crimes = LawArticleRetriever._extract_crimes(snippet)
            if len(snippet) > max_chars:
                snippet = snippet[:max_chars] + "..."
            crime_text = f" [{'；'.join(crimes)}]" if crimes else ""
            lines.append(f"- Article {law_id}{crime_text}: {snippet}")
        return "\n".join(lines)
