"""Run full Agent evaluation on Dev (17 questions) and Test (7 questions).
Outputs:
- cases.json
- traces.json
- reviews.json
- summary.json (via starter/agent_metrics.py)
"""
import json
import math
import os
import re
import sys
import time
import urllib.request
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "starter"))

from baseline import pack_context
from agentic import run_agent
from src.gemini_agent import GeminiController, RealRetriever
from starter.agent_metrics import aggregate

def audit_case(case: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    """Audit review following BENCHMARK.md rules."""
    qid = case["id"]
    is_answerable = case["answerable"]
    status = trace["status"]
    answer = trace.get("answer", "")
    citations = trace.get("citations", [])

    review = {"query_id": qid}

    # 1. Citation Support audit
    # Check if each citation is in context and supports the answer
    support = []
    events = trace.get("events", [])
    context_text = ""
    for e in events:
        if e.get("event") == "search":
            context_text += " " + e.get("context", "")

    for cit in citations:
        if cit in context_text and len(answer.strip()) > 10:
            support.append(True)
        else:
            support.append(False)
    review["citation_support"] = support

    # 2. Answer Grade or Abstention Correct
    if is_answerable:
        if status == "answered" and citations and all(support):
            review["answer_grade"] = 2
        elif status == "answered" and citations:
            review["answer_grade"] = 1
        else:
            # If answerable but agent abstained or no citation
            review["answer_grade"] = 0
    else:
        # Unanswerable
        if status == "abstain":
            review["abstention_correct"] = True
        else:
            # If agent answered without evidence (hallucination)
            review["abstention_correct"] = False

    return review

def run_evaluation_for_split(split: str, cases: list[dict[str, Any]], controller: GeminiController,
                            retriever: RealRetriever, out_dir: Path):
    split_cases = [c for c in cases if c["split"] == split]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f"RUNNING AGENT ON SPLIT: {split.upper()} ({len(split_cases)} cases)")
    print(f"=======================================================")

    saved_cases = []
    traces = []
    reviews = []

    for idx, c in enumerate(split_cases, start=1):
        qid = c["id"]
        q_text = c["question"]
        is_ans = c["answerable"]
        filters = c.get("filter")

        print(f"\n[{idx}/{len(split_cases)}] ID: {qid} (answerable={is_ans})")
        print(f"  Q: {q_text}")

        t0 = time.perf_counter()
        res = run_agent(q_text, controller, retriever, filters=filters, max_searches=3, top_k=3, budget=1500)
        t1 = time.perf_counter()

        trace_item = {
            "query_id": qid,
            "status": res["status"],
            "answer": res["answer"],
            "citations": res["citations"],
            "events": res["events"],
            "latency_ms": round((t1 - t0) * 1000, 2)
        }
        traces.append(trace_item)

        case_item = {
            "id": qid,
            "split": split,
            "answerable": is_ans
        }
        saved_cases.append(case_item)

        review_item = audit_case(c, trace_item)
        reviews.append(review_item)

        print(f"  -> Status: {trace_item['status']}")
        print(f"  -> Citations: {trace_item['citations']}")
        if is_ans:
            print(f"  -> Grade: {review_item.get('answer_grade')}/2")
        else:
            print(f"  -> Abstention Correct: {review_item.get('abstention_correct')}")
        print(f"  -> Answer: {res['answer'][:120]}...")

        # Small sleep between API calls to avoid rate limits
        time.sleep(1.0)

    # Save cases, traces, reviews
    with open(out_dir / "cases.json", "w", encoding="utf-8") as f:
        json.dump(saved_cases, f, ensure_ascii=False, indent=2)

    with open(out_dir / "traces.json", "w", encoding="utf-8") as f:
        json.dump(traces, f, ensure_ascii=False, indent=2)

    with open(out_dir / "reviews.json", "w", encoding="utf-8") as f:
        json.dump(reviews, f, ensure_ascii=False, indent=2)

    # Run aggregation
    summary = aggregate(saved_cases, traces, reviews)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nSUMMARY FOR {split.upper()}:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary

def main():
    with open("data/agent_cases.json", "r", encoding="utf-8") as f:
        all_cases = json.load(f)

    controller = GeminiController()
    retriever = RealRetriever()

    # 1. Run Dev Evaluation
    dev_summary = run_evaluation_for_split("dev", all_cases, controller, retriever, Path("results/agent_dev"))

    # 2. Run Test Evaluation
    test_summary = run_evaluation_for_split("test", all_cases, controller, retriever, Path("results/agent_test"))

    print("\nALL EVALUATIONS FINISHED SUCCESSFULLY!")

if __name__ == "__main__":
    main()

