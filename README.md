# text-kmeans-lite

纯 Python 实现的 **TF-IDF + k-means 文本聚类**。无需 numpy / sklearn，零第三方依赖。

## 功能

- 零依赖 TF-IDF 向量化（稀疏 dict）；
- k-means++ 与随机两种初始化；
- 余弦距离 / 欧氏距离可切换；
- 迭代至分配稳定即收敛；
- 输出每簇文档分组。

## 快速开始

```bash
python3 cli.py -k 2 \
  "足球 篮球 比赛 运动员 冠军" \
  "联赛 球队 进球 球迷" \
  "人工智能 算法 芯片 编程 数据" \
  "机器学习 软件 模型 计算机"
```

## 使用示例

```bash
# 欧氏距离
python3 cli.py -k 2 --distance euclidean "doc one" "doc two" "doc three"
```

## 无 API Key 如何运行

本工具**完全不需要 API Key**，聚类为本地数值计算。

## 目录结构

```
text-kmeans-lite/
├── kmeans.py     # 向量化、距离、kmeans、聚类接口
├── cli.py        # 命令行入口
├── tests/
│   └── test_kmeans.py
├── README.md
├── LICENSE
└── .gitignore
```

## 测试

```bash
python3 -m unittest discover -s tests
```

## 许可证

[MIT](./LICENSE)
