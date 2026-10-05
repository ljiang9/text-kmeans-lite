"""命令行：python3 cli.py k doc1 doc2 ..."""
import argparse
import json

from kmeans import cluster_documents, TfidfVectorizer, KMeans


def main(argv=None):
    p = argparse.ArgumentParser(description="text-kmeans-lite 文本聚类")
    p.add_argument("-k", type=int, default=2, help="簇数")
    p.add_argument("--distance", choices=["cosine", "euclidean"], default="cosine")
    p.add_argument("docs", nargs="+", help="待聚类文档")
    args = p.parse_args(argv)

    groups = cluster_documents(args.docs, k=args.k, distance=args.distance)
    result = [
        {"cluster": i, "docs": [args.docs[j] for j in g]}
        for i, g in enumerate(groups)
    ]
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
