"""Export Strategy B chunks from Day 6 PostgreSQL to data/day6_chunks.json."""
import json
from pathlib import Path
import psycopg

def main():
    db_url = "postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab"
    output_path = Path("data/day6_chunks.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT chunk_id, document_id, source, page, content, metadata
                FROM document_chunks
                WHERE strategy = 'B'
                ORDER BY page, chunk_id;
            """)
            rows = cur.fetchall()

    chunks = []
    for r in rows:
        chunk = {
            "id": r[0],
            "document_id": r[1],
            "source": r[2],
            "source_id": r[1],
            "page": r[3],
            "text": r[4],
            "metadata": r[5]
        }
        chunks.append(chunk)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Successfully exported {len(chunks)} chunks to {output_path}")

if __name__ == "__main__":
    main()

