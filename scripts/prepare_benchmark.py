"""Fix evidence quotes so that they match chunk['text'] exactly as literal substrings."""
import json
import re
import unicodedata
from pathlib import Path

def normalize_str(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def find_exact_substring(chunk_text: str, quote: str) -> str | None:
    # 1. Exact match
    if quote in chunk_text:
        return quote
        
    # 2. Try replacing whitespace with \s+ in regex
    words = quote.split()
    pattern = r"\s+".join(re.escape(w) for w in words)
    match = re.search(pattern, chunk_text)
    if match:
        return match.group(0)
        
    # 3. Try with first 6 and last 4 words if quote had minor edits
    if len(words) >= 6:
        pattern = r"\s+".join(re.escape(w) for w in words[:6])
        match = re.search(pattern, chunk_text)
        if match:
            pattern_end = r"\s+".join(re.escape(w) for w in words[-4:])
            match_end = re.search(pattern_end, chunk_text[match.start():])
            if match_end:
                return chunk_text[match.start() : match.start() + match_end.end()]
            return match.group(0)
            
    return None

def main():
    with open("../K5-L3A-DAY06-NguyenNgocHieu-2A202603502-DataIngestion/benchmark/benchmark_questions.json", "r", encoding="utf-8") as f:
        day6_q = json.load(f)
    with open("data/day6_chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    retrieval_questions = []
    
    for q in day6_q:
        page = q["gold_page"]
        source_id = q["gold_source_id"]
        quote = q["evidence_quote"]
        
        # Candidate chunks on this page
        page_chunks = [c for c in chunks if c["page"] == page and c["source_id"] == source_id]
        
        matched_chunk = None
        exact_sub = None
        
        for c in page_chunks:
            exact_sub = find_exact_substring(c["text"], quote)
            if exact_sub:
                matched_chunk = c
                break
                
        if not exact_sub:
            for c in chunks:
                exact_sub = find_exact_substring(c["text"], quote)
                if exact_sub:
                    matched_chunk = c
                    page = c["page"]
                    break
                    
        if not exact_sub:
            raise ValueError(f"Could not locate exact evidence for question {q['id']}: {quote}")
            
        print(f"[{q['id']}] Found exact evidence in chunk {matched_chunk['id']} (page {matched_chunk['page']}) length {len(exact_sub)}")
        
        rq = {
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "gold": [
                {
                    "source_id": matched_chunk["source_id"],
                    "page": matched_chunk["page"],
                    "evidence": exact_sub
                }
            ]
        }
        if q.get("filter"):
            rq["filter"] = q["filter"]
        retrieval_questions.append(rq)
        
    for q in retrieval_questions:
        g = q["gold"][0]
        hits = [c for c in chunks if c["source_id"] == g["source_id"] and c["page"] == g["page"] and g["evidence"] in c["text"]]
        if not hits:
            raise ValueError(f"Strict hit failed for {q['id']}")
            
    print(f"ALL {len(retrieval_questions)} questions PASS STRICT BASELINE HIT!")
    
    with open("data/questions.json", "w", encoding="utf-8") as f:
        json.dump(retrieval_questions, f, ensure_ascii=False, indent=2)

    unanswerable_questions = [
        {
            "id": "q_dev_unans_01",
            "split": "dev",
            "question": "Mức xử phạt tiền đối với hành vi điều khiển xe máy vượt đèn đỏ là bao nhiêu triệu đồng theo quy định của Luật Đường bộ?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Luật Đường bộ 2024 chỉ quy định về kết cấu hạ tầng và vận tải; mức xử phạt vi phạm hành chính (như vượt đèn đỏ) thuộc Nghị định xử phạt vi phạm hành chính trong lĩnh vực giao thông đường bộ."
        },
        {
            "id": "q_dev_unans_02",
            "split": "dev",
            "question": "Thời hạn bảo hành đối với công trình xây dựng đường sắt đô thị kết nối đường bộ được quy định tối thiểu là bao nhiêu năm?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Luật Đường bộ 2024 không quy định về thời hạn bảo hành công trình đường sắt đô thị (thuộc phạm vi Luật Đường sắt và pháp luật xây dựng)."
        },
        {
            "id": "q_test_unans_01",
            "split": "test",
            "question": "Quy trình cấp Giấy phép lái xe quốc tế (IDP) cho công dân Việt Nam gồm những bước và hồ sơ cụ thể nào?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Luật Đường bộ 2024 không quy định chi tiết hồ sơ thủ tục cấp Giấy phép lái xe quốc tế; nội dung này thuộc Thông tư của Bộ GTVT và Luật Trật tự, an toàn giao thông đường bộ."
        },
        {
            "id": "q_test_unans_02",
            "split": "test",
            "question": "Mức phí bảo hiểm trách nhiệm dân sự bắt buộc hàng năm đối với xe ô tô chở người dưới 6 chỗ ngồi không kinh doanh vận tải là bao nhiêu tiền?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Phí bảo hiểm bắt buộc trách nhiệm dân sự của chủ xe cơ giới do Chính phủ quy định tại Nghị định riêng về bảo hiểm bắt buộc, không quy định trong Luật Đường bộ 2024."
        }
    ]
    
    agent_cases = []
    for q in retrieval_questions:
        agent_cases.append({
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "answerable": True,
            "expected_behavior": "search_and_answer",
            "expected_answer": q["expected_answer"],
            "gold": q["gold"],
            "filter": q.get("filter")
        })
    agent_cases.extend(unanswerable_questions)
    agent_cases.sort(key=lambda x: (x["split"], x["id"]))
    
    with open("data/agent_cases.json", "w", encoding="utf-8") as f:
        json.dump(agent_cases, f, ensure_ascii=False, indent=2)
        
    print(f"Created data/agent_cases.json with {len(agent_cases)} questions (17 dev, 7 test).")

if __name__ == "__main__":
    main()
