"""text-kmeans-lite — TF-IDF + kmeans 文本聚类（纯 Python）。

纯 Python 实现：词频/逆文档频率向量化、k-means++ 与随机初始化、
余弦/欧氏距离、迭代至收敛，输出每簇文档。零第三方依赖。
"""
from __future__ import annotations

import math
import random
import re
from collections import Counter

_EN_RE = re.compile(r"[a-zA-Z]+")


def tokenize(text: str) -> list[str]:
    toks = _EN_RE.findall(text.lower())
    toks += re.findall(r"[\u4e00-\u9fff]", text)
    return toks


class TfidfVectorizer:
    """把文档集合转为稀疏 TF-IDF 向量（dict: term -> weight）。"""

    def fit_transform(self, docs: list[str]) -> list[dict]:
        doc_tokens = [tokenize(d) for d in docs]
        n = max(1, len(doc_tokens))
        df: Counter = Counter()
        for toks in doc_tokens:
            for t in set(toks):
                df[t] += 1
        self.idf_ = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}

        vectors = []
        for toks in doc_tokens:
            tf = Counter(toks)
            total = max(1, len(toks))
            vec = {}
            for t, c in tf.items():
                vec[t] = (c / total) * self.idf_.get(t, 1.0)
            vectors.append(vec)
        return vectors


def _norm(vec: dict) -> float:
    return math.sqrt(sum(v * v for v in vec.values()))


def cosine_distance(a: dict, b: dict) -> float:
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    na, nb = _norm(a), _norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - dot / (na * nb)


def euclidean_distance(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    s = sum((a.get(k, 0.0) - b.get(k, 0.0)) ** 2 for k in keys)
    return math.sqrt(s)


class KMeans:
    def __init__(self, k: int, init: str = "kmeans++",
                 distance: str = "cosine", max_iter: int = 100, seed: int = 42):
        self.k = k
        self.init = init
        self.dist = cosine_distance if distance == "cosine" else euclidean_distance
        self.max_iter = max_iter
        self.rng = random.Random(seed)
        self.centroids_: list[dict] = []
        self.labels_: list[int] = []

    def _pick_init(self, vectors: list[dict]):
        if self.init == "random":
            return [dict(v) for v in self.rng.sample(vectors, self.k)]
        first = self.rng.choice(vectors)
        centroids = [dict(first)]
        while len(centroids) < self.k:
            dists = [min(self.dist(v, c) ** 2 for c in centroids) for v in vectors]
            total = sum(dists) or 1.0
            r = self.rng.random() * total
            acc = 0.0
            chosen = vectors[-1]
            for v, d in zip(vectors, dists):
                acc += d
                if acc >= r:
                    chosen = v
                    break
            centroids.append(dict(chosen))
        return centroids

    def _mean(self, members: list[dict]) -> dict:
        if not members:
            return {}
        out: dict[str, float] = {}
        for m in members:
            for t, v in m.items():
                out[t] = out.get(t, 0.0) + v
        return {t: v / len(members) for t, v in out.items()}

    def fit(self, vectors: list[dict]):
        n = len(vectors)
        if self.k > n:
            raise ValueError("k 不能大于文档数")
        self.centroids_ = self._pick_init(vectors)
        labels = [0] * n
        for _ in range(self.max_iter):
            new_labels = []
            for v in vectors:
                best = min(range(self.k),
                           key=lambda c: self.dist(v, self.centroids_[c]))
                new_labels.append(best)
            if new_labels == labels:
                break
            labels = new_labels
            for c in range(self.k):
                members = [vectors[i] for i in range(n) if labels[i] == c]
                self.centroids_[c] = self._mean(members)
        self.labels_ = labels
        return self


def cluster_documents(docs: list[str], k: int = 2, **kwargs) -> list[list[int]]:
    """对文档聚类，返回每簇的文档下标列表。"""
    vec = TfidfVectorizer().fit_transform(docs)
    km = KMeans(k=k, **kwargs).fit(vec)
    groups: list[list[int]] = [[] for _ in range(k)]
    for idx, label in enumerate(km.labels_):
        groups[label].append(idx)
    return [g for g in groups if g]
