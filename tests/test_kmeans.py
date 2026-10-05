"""text-kmeans-lite 单元测试。运行：python3 -m unittest discover -s tests"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kmeans import (  # noqa: E402
    TfidfVectorizer, KMeans, cluster_documents,
    cosine_distance, euclidean_distance,
)


class TestVectorizer(unittest.TestCase):
    def test_shape(self):
        vecs = TfidfVectorizer().fit_transform(["hello world", "hello there"])
        self.assertEqual(len(vecs), 2)
        self.assertIn("hello", vecs[0])


class TestDistance(unittest.TestCase):
    def test_cosine_identical(self):
        v = {"a": 1.0, "b": 2.0}
        self.assertAlmostEqual(cosine_distance(v, dict(v)), 0.0, places=4)

    def test_euclidean(self):
        self.assertAlmostEqual(
            euclidean_distance({"a": 3.0}, {"a": 0.0}), 3.0, places=4)


class TestKMeans(unittest.TestCase):
    def test_separates_two_topics(self):
        docs = [
            "足球 篮球 比赛 运动员 冠军",
            "联赛 球队 进球 球迷",
            "人工智能 算法 芯片 编程 数据",
            "机器学习 软件 模型 计算机",
        ]
        groups = cluster_documents(docs, k=2, seed=1)
        self.assertEqual(len(groups), 2)
        for g in groups:
            self.assertEqual(len(g), 2)
        flat = {doc: label for g, label in zip(groups, range(2)) for doc in g}
        self.assertEqual(flat[0], flat[1])
        self.assertEqual(flat[2], flat[3])
        self.assertNotEqual(flat[0], flat[2])

    def test_random_init(self):
        docs = ["a b c", "a b", "x y z", "x y"]
        groups = cluster_documents(docs, k=2, init="random", seed=7)
        self.assertEqual(len(groups), 2)

    def test_k_exceeds_docs(self):
        vec = TfidfVectorizer().fit_transform(["only one"])
        with self.assertRaises(ValueError):
            KMeans(k=3).fit(vec)


if __name__ == "__main__":
    unittest.main()
