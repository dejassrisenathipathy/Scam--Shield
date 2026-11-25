import csv, json, random, collections

random.seed(42)  # same seed = same 1,000 messages every time
rows = list(csv.DictReader(open("data/sms_test_public.csv", encoding="utf-8")))
safe = [r for r in rows if r["label"].strip() in ("0", "0.0")]
scam = [r for r in rows if r["label"].strip() in ("1", "1.0")]
print(f"Available: {len(safe)} genuine, {len(scam)} scam")

N_SAFE, N_SCAM = 800, 200  # 20% scams, like a real inbox
assert len(safe) >= N_SAFE and len(scam) >= N_SCAM, "Not enough messages"

picked = random.sample(safe, N_SAFE) + random.sample(scam, N_SCAM)
random.shuffle(picked)
out = [{"id": i + 1, "text": r["text"], "label": int(float(r["label"])),
        "category": r["category"], "source": r["source"]} for i, r in enumerate(picked)]

with open("data/test_1000.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
with open("data/test_1000.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["id", "text", "label", "category", "source"])
    w.writeheader()
    w.writerows(out)

print("Wrote 1,000 messages to data/test_1000.json and data/test_1000.csv")
print("Labels (0 = genuine, 1 = scam):", dict(collections.Counter(o["label"] for o in out)))
print("Sources:", dict(collections.Counter(o["source"] for o in out)))
