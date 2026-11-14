"""Retrieval methods and benchmarking script for Day 7.

Implements:
1. TF-IDF Lexical Retrieval
2. Dense Retrieval (pgvector + google/embeddinggemma-2)
3. Hybrid Retrieval (Dense + Lexical with Reciprocal Rank Fusion - RRF)
"""
import argparse
import json
import math
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any
import psycopg
import torch
from sentence_transformers import SentenceTransformer

# -------------------------------------------------------------
# 1. Text Tokenization & Utilities
# -------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())

def pack_context(chunks: list[dict[str, Any]], budget: int = 1500) -> tuple[str, list[str]]:
    """Pack chunks into context string within character budget."""
    seen_ids = set()
    kept_chunks = []
    used_chars = 0
    for c in chunks:
        cid = c["id"]
        if cid in seen_ids:
            continue
        text = c["text"].strip()
        add_chars = len(text) + 2
        if used_chars + add_chars > budget and kept_chunks:
            break
        kept_chunks.append(c)
        seen_ids.add(cid)
        used_chars += add_chars

    parts = [f"[{c['id']}] {c['text'].strip()}" for c in kept_chunks]
    return "\n\n".join(parts), [c["id"] for c in kept_chunks]

def check_hit(chunk: dict[str, Any], gold: list[dict[str, Any]]) -> bool:
    return any(
        chunk["source_id"] == g["source_id"] and chunk["page"] == g["page"] and g["evidence"] in chunk["text"]
        for g in gold
    )

# -------------------------------------------------------------
# 2. Retrievers
# -------------------------------------------------------------
class TfidfRetriever:
    def __init__(self, chunks: list[dict[str, Any]]):
        self.chunks = chunks
        counts = [Counter(tokenize(c["text"])) for c in chunks]
        df = Counter(term for row in counts for term in row)
        self.idf = {term: math.log((1 + len(chunks)) / (1 + count)) + 1 for term, count in df.items()}
        self.vectors = [self._vector(row) for row in counts]

    def _vector(self, counts: Counter) -> dict[str, float]:
        vector = {term: count * self.idf[term] for term, count in counts.items() if term in self.idf}
        norm = math.sqrt(sum(v * v for v in vector.values())) or 1.0
        return {term: v / norm for term, v in vector.items()}

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        q_vec = self._vector(Counter(tokenize(query)))
        scores = []
        for idx, c in enumerate(self.chunks):
            if filters:
                if any(str(c["metadata"].get(k, "")).lower() != str(v).lower() for k, v in filters.items()):
                    continue
            d_vec = self.vectors[idx]
            dot = sum(w * d_vec.get(term, 0.0) for term, w in q_vec.items())
            scores.append((dot, idx))
        scores.sort(key=lambda x: x[0], reverse=True)
        results = []
        for rank, (score, idx) in enumerate(scores[:top_k], start=1):
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            chunk["rank"] = rank
            results.append(chunk)
        return results


class DenseRetriever:
    def __init__(self, db_url: str, model_name: str = "google/embeddinggemma-2"):
        self.db_url = db_url
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs = {"dtype": torch.float32} if device == "cpu" else {}
        self.model = SentenceTransformer(model_name, device=device, model_kwargs=model_kwargs)

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        q_emb = self.model.encode(query, normalize_embeddings=True).tolist()
        vec_str = "[" + ",".join(f"{x:.6f}" for x in q_emb) + "]"

        sql = """
            SELECT chunk_id, document_id, source, page, content, metadata,
                   1 - (embedding <=> %s::vector) AS score
            FROM document_chunks
            WHERE strategy = 'B'
        """
        params: list[Any] = [vec_str]
        if filters:
            for k, v in filters.items():
                sql += " AND LOWER(metadata->>%s) = LOWER(%s)"
                params.extend([k, str(v)])

        sql += " ORDER BY embedding <=> %s::vector ASC LIMIT %s;"
        params.extend([vec_str, top_k])

        with psycopg.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        results = []
        for rank, r in enumerate(rows, start=1):
            results.append({
                "id": r[0],
                "document_id": r[1],
                "source": r[2],
                "source_id": r[1],
                "page": r[3],
                "text": r[4],
                "metadata": r[5],
                "score": float(r[6]),
                "rank": rank
            })
        return results


class HybridRetriever:
    def __init__(self, dense_retriever: DenseRetriever, tfidf_retriever: TfidfRetriever, rrf_k: int = 60):
        self.dense = dense_retriever
        self.tfidf = tfidf_retriever
        self.rrf_k = rrf_k

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3, candidate_k: int = 10) -> list[dict[str, Any]]:
        dense_hits = self.dense.search(query, filters=filters, top_k=candidate_k)
        tfidf_hits = self.tfidf.search(query, filters=filters, top_k=candidate_k)

        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, dict[str, Any]] = {}

        for rank, h in enumerate(dense_hits, start=1):
            cid = h["id"]
            chunk_map[cid] = h
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        for rank, h in enumerate(tfidf_hits, start=1):
            cid = h["id"]
            if cid not in chunk_map:
                chunk_map[cid] = h
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for rank, (cid, score) in enumerate(ranked[:top_k], start=1):
            chunk = dict(chunk_map[cid])
            chunk["score"] = float(score)
            chunk["rank"] = rank
            results.append(chunk)
        return results


# -------------------------------------------------------------
# 3. Evaluation Runner
# -------------------------------------------------------------
def evaluate_retriever(name: str, retriever, questions: list[dict[str, Any]], split: str,
                       budget: int = 1500, top_k: int = 3) -> dict[str, Any]:
    filtered_q = [q for q in questions if q.get("split") == split]
    if not filtered_q:
        raise ValueError(f"Không có câu hỏi nào cho split: {split}")

    total = len(filtered_q)
    hit1_count = 0
    hit3_count = 0
    context_hit_count = 0
    latencies = []
    traces = []

    for q in filtered_q:
        filters = q.get("filter")
        t0 = time.perf_counter()
        hits = retriever.search(q["question"], filters=filters, top_k=top_k)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

        gold = q["gold"]
        h1 = check_hit(hits[0], gold) if hits else False
        h3 = any(check_hit(h, gold) for h in hits[:3])

        context, context_ids = pack_context(hits, budget=budget)
        ctx_hit = any(check_hit(h, gold) for h in hits if h["id"] in context_ids)

        if h1:
            hit1_count += 1
        if h3:
            hit3_count += 1
        if ctx_hit:
            context_hit_count += 1

        traces.append({
            "question_id": q["id"],
            "question": q["question"],
            "hit_at_1": h1,
            "hit_at_3": h3,
            "context_evidence_hit": ctx_hit,
            "top_hits": [
                {"rank": h["rank"], "id": h["id"], "page": h["page"], "score": h["score"], "is_hit": check_hit(h, gold)}
                for h in hits
            ],
            "context_ids": context_ids
        })

    latencies.sort()
    med_latency = latencies[len(latencies) // 2]

    summary = {
        "method": name,
        "split": split,
        "total_queries": total,
        "hit1": hit1_count,
        "hit1_pct": round(hit1_count / total * 100, 2),
        "hit1_ratio": f"{hit1_count}/{total}",
        "hit3": hit3_count,
        "hit3_pct": round(hit3_count / total * 100, 2),
        "hit3_ratio": f"{hit3_count}/{total}",
        "context_evidence_rate": round(context_hit_count / total * 100, 2),
        "context_evidence_ratio": f"{context_hit_count}/{total}",
        "context_budget_chars": budget,
        "median_latency_ms": round(med_latency, 2),
        "cost_usd": 0.0
    }

    return {"summary": summary, "traces": traces}


def main():
    parser = argparse.ArgumentParser(description="Evaluate 3 retrieval methods on Dev")
    parser.add_argument("--chunks", default="data/day6_chunks.json")
    parser.add_argument("--questions", default="data/questions.json")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--budget", type=int, default=1500)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--output_dir", default="results/retrieval_comparison")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.chunks, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(args.questions, "r", encoding="utf-8") as f:
        questions = json.load(f)

    db_url = "postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab"

    print(f"Initializing Retrievers on {len(chunks)} chunks...")
    tfidf = TfidfRetriever(chunks)
    dense = DenseRetriever(db_url=db_url)
    hybrid = HybridRetriever(dense, tfidf, rrf_k=60)

    methods = [
        ("TF-IDF Lexical", tfidf),
        ("Dense (pgvector + embeddinggemma-2)", dense),
        ("Hybrid (Dense + Lexical RRF)", hybrid)
    ]

    all_summaries = []

    for name, ret in methods:
        print(f"\nEvaluating: {name} on {args.split} (budget={args.budget}, top_k={args.top_k})...")
        res = evaluate_retriever(name, ret, questions, split=args.split, budget=args.budget, top_k=args.top_k)
        summary = res["summary"]
        all_summaries.append(summary)
        print(f"  -> Hit@1: {summary['hit1_ratio']} ({summary['hit1_pct']}%)")
        print(f"  -> Hit@3: {summary['hit3_ratio']} ({summary['hit3_pct']}%)")
        print(f"  -> Context Evidence Rate: {summary['context_evidence_ratio']} ({summary['context_evidence_rate']}%)")
        print(f"  -> Median Latency: {summary['median_latency_ms']} ms")

        # Save trace
        slug = re.sub(r"[^\w]+", "_", name.lower()).strip("_")
        with open(out_dir / f"{slug}_{args.split}_traces.json", "w", encoding="utf-8") as f:
            json.dump(res["traces"], f, ensure_ascii=False, indent=2)

    with open(out_dir / f"summary_{args.split}.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    print(f"\nSaved all results to {out_dir}/summary_{args.split}.json")

if __name__ == "__main__":
    main()

