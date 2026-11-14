"""Evaluate and compare 3 retrieval methods:
1. TF-IDF Lexical (live computed)
2. Dense Retrieval (pgvector + google/embeddinggemma-2 from Day 6 Strategy B)
3. Hybrid Retrieval (RRF combining Dense + TF-IDF)
"""
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())

def pack_context(chunks: list[dict[str, Any]], budget: int = 1500) -> tuple[str, list[str]]:
    seen_ids = set()
    kept_chunks = []
    used_chars = 0
    for c in chunks:
        cid = c["id"] if "id" in c else c["chunk_id"]
        if cid in seen_ids:
            continue
        text = c["text"].strip()
        add_chars = len(text) + 2
        if used_chars + add_chars > budget and kept_chunks:
            break
        kept_chunks.append(c)
        seen_ids.add(cid)
        used_chars += add_chars
    ids = [c["id"] if "id" in c else c["chunk_id"] for c in kept_chunks]
    parts = [f"[{cid}] {c['text'].strip()}" for cid, c in zip(ids, kept_chunks)]
    return "\n\n".join(parts), ids

def check_hit(chunk: dict[str, Any], gold: list[dict[str, Any]]) -> bool:
    c_source = chunk.get("source_id", chunk.get("document_id"))
    c_page = chunk.get("page")
    c_text = chunk.get("text", "")
    return any(
        c_source == g["source_id"] and c_page == g["page"] and g["evidence"] in c_text
        for g in gold
    )

def main():
    with open("data/day6_chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open("data/questions.json", "r", encoding="utf-8") as f:
        questions = json.load(f)
    with open("../K5-L3A-DAY06-NguyenNgocHieu-2A202603502-DataIngestion/starter/results/strategy_B_traces.json", "r", encoding="utf-8") as f:
        day6_dense_traces = json.load(f)

    dense_trace_map = {t["question_id"]: t for t in day6_dense_traces}

    # 1. TF-IDF Retriever
    counts = [Counter(tokenize(c["text"])) for c in chunks]
    df = Counter(term for row in counts for term in row)
    idf = {term: math.log((1 + len(chunks)) / (1 + count)) + 1 for term, count in df.items()}
    doc_vectors = []
    for row in counts:
        vec = {term: count * idf[term] for term, count in row.items() if term in idf}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        doc_vectors.append({term: v / norm for term, v in vec.items()})

    def search_tfidf(query: str, filters: dict[str, Any] | None = None, top_k: int = 10):
        q_vec = {term: count * idf[term] for term, count in Counter(tokenize(query)).items() if term in idf}
        norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
        q_vec = {term: v / norm for term, v in q_vec.items()}
        scores = []
        for idx, c in enumerate(chunks):
            if filters:
                if any(str(c["metadata"].get(k, "")).lower() != str(v).lower() for k, v in filters.items()):
                    continue
            dot = sum(w * doc_vectors[idx].get(term, 0.0) for term, w in q_vec.items())
            scores.append((dot, idx))
        scores.sort(key=lambda x: x[0], reverse=True)
        res = []
        for rank, (score, idx) in enumerate(scores[:top_k], start=1):
            item = dict(chunks[idx])
            item["score"] = float(score)
            item["rank"] = rank
            res.append(item)
        return res

    for split in ["dev", "test"]:
        split_q = [q for q in questions if q["split"] == split]
        n = len(split_q)
        print(f"\n==================== SPLIT: {split.upper()} (N = {n}) ====================")

        for budget in [800, 1500, 2500]:
            print(f"\n--- Context Budget: {budget} chars ---")
            for method_name in ["TF-IDF Lexical", "Dense (pgvector)", "Hybrid (RRF Dense + TF-IDF)"]:
                h1_cnt = 0
                h3_cnt = 0
                ctx_cnt = 0

                for q in split_q:
                    gold = q["gold"]
                    filters = q.get("filter")
                    qid = q["id"]

                    # Hits selection
                    if method_name == "TF-IDF Lexical":
                        hits = search_tfidf(q["question"], filters=filters, top_k=3)
                    elif method_name == "Dense (pgvector)":
                        dense_t = dense_trace_map[qid]
                        hits = dense_t["top_hits"]
                        # Map chunk fields
                        for h in hits:
                            h["source_id"] = "luat_duong_bo_2024"
                    else: # Hybrid
                        tfidf_hits = search_tfidf(q["question"], filters=filters, top_k=5)
                        dense_hits = dense_trace_map[qid]["top_hits"]
                        for h in dense_hits:
                            h["source_id"] = "luat_duong_bo_2024"

                        rrf_scores = {}
                        chunk_obj_map = {}
                        for rank, h in enumerate(dense_hits, start=1):
                            cid = h["chunk_id"]
                            chunk_obj_map[cid] = h
                            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (60 + rank))
                        for rank, h in enumerate(tfidf_hits, start=1):
                            cid = h["id"]
                            chunk_obj_map[cid] = h
                            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (60 + rank))

                        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
                        hits = [chunk_obj_map[cid] for cid, _ in sorted_rrf[:3]]

                    # Evaluate metrics
                    h1 = check_hit(hits[0], gold) if hits else False
                    h3 = any(check_hit(h, gold) for h in hits[:3])
                    _, ctx_ids = pack_context(hits, budget=budget)
                    ctx_hit = any(check_hit(h, gold) for h in hits if (h.get("id") or h.get("chunk_id")) in ctx_ids)

                    if h1: h1_cnt += 1
                    if h3: h3_cnt += 1
                    if ctx_hit: ctx_cnt += 1

                print(f"[{method_name:<26}] Hit@1: {h1_cnt:2d}/{n} ({h1_cnt/n*100:5.1f}%) | Hit@3: {h3_cnt:2d}/{n} ({h3_cnt/n*100:5.1f}%) | Context Ev.: {ctx_cnt:2d}/{n} ({ctx_cnt/n*100:5.1f}%)")

if __name__ == "__main__":
    main()

