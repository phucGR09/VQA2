import json
import statistics
from transformers import AutoTokenizer

DB_PATH = "./data/merged_7_database.json"
MODEL_NAME = "intfloat/multilingual-e5-large"

tok = AutoTokenizer.from_pretrained(MODEL_NAME)

with open(DB_PATH) as f:
    db = json.load(f)

token_counts = []
for article_id, article in db.items():
    content = article.get("content", "")
    n = len(tok(content, truncation=False)["input_ids"])
    token_counts.append((article_id, n))

counts = [n for _, n in token_counts]
token_counts.sort(key=lambda x: x[1], reverse=True)

print(f"Total articles : {len(counts)}")
print(f"Max tokens     : {max(counts)}")
print(f"Min tokens     : {min(counts)}")
print(f"Mean tokens    : {statistics.mean(counts):.0f}")
print(f"Median tokens  : {statistics.median(counts):.0f}")
print(f"P90 tokens     : {sorted(counts)[int(0.9 * len(counts))]}")
print(f"P95 tokens     : {sorted(counts)[int(0.95 * len(counts))]}")
print()
print("Overflow ratio (content > 512 tokens):")
for threshold in [512, 1024, 4096, 8192]:
    over = sum(1 for n in counts if n > threshold)
    print(f"  > {threshold:>5} tokens: {over:>5} articles ({100*over/len(counts):.1f}%)")
print()
print("Top 5 longest articles:")
for article_id, n in token_counts[:5]:
    print(f"  {article_id}: {n} tokens")
