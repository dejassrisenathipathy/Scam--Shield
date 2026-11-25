"""Evaluate the 4 SMS .pte models on real messages, exactly as the phone does.
Run in the SMS environment (ExecuTorch 1.6.0) from ~/scam_shield:
    python ml/eval_real.py data/real_test_463.csv
"""
import csv, json, os, re, sys, unicodedata
import torch
from executorch.runtime import Runtime

ROOT = os.path.expanduser("~/scam_shield")
def find(name):
    for d, dirs, files in os.walk(ROOT):
        dirs[:] = [x for x in dirs if x not in {"raw", ".git", "__pycache__", "models_v0", "v0"}]
        if name in files: return os.path.join(d, name)
    sys.exit(f"Can't find {name} under {ROOT}")

V = json.load(open(find("vocab.json"), encoding="utf-8"))
MAX, PAD, UNK, CH = V["max_len"], V["pad_id"], V["unk_id"], V["chars"]

def encode(text):
    t = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()
    ids = [CH.get(c, UNK) for c in t[:MAX]]
    return ids + [PAD] * (MAX - len(ids))

# 1) Prove our encoding matches the one used for the phone
vec = json.load(open(find("test_vectors.json"), encoding="utf-8"))
bad = sum(encode(v["text"]) != v["input_ids"] for v in vec)
print(f"Encoding check: {len(vec) - bad}/{len(vec)} test vectors match")
if bad: sys.exit("Encoding differs from the phone's. Stop and send this to Claude.")

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
y = [int(r["label"]) for r in rows]
X = [torch.tensor([encode(r["text"])], dtype=torch.long) for r in rows]
rt = Runtime.get()

def metrics(p, y, th):
    tp = sum(1 for a, b in zip(p, y) if a >= th and b == 1); fp = sum(1 for a, b in zip(p, y) if a >= th and b == 0)
    fn = sum(1 for a, b in zip(p, y) if a < th and b == 1); tn = sum(1 for a, b in zip(p, y) if a < th and b == 0)
    prec = tp / (tp + fp) if tp + fp else 0; rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    return dict(acc=(tp + tn) / len(y), prec=prec, rec=rec, f1=f1, fa=fp / (fp + tn) if fp + tn else 0, fp=fp, fn=fn)

os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
summary = {}
print(f"\n{len(rows)} real messages ({sum(y)} scam, {len(y) - sum(y)} genuine), threshold 0.5\n")
print(f"{'model':16} {'acc':>6} {'prec':>6} {'recall':>6} {'F1':>6} {'false alarms':>13} {'missed':>7}")
for name in ["sms_tiny_fp32", "sms_tiny_int8", "sms_deep_fp32", "sms_deep_int8"]:
    m = rt.load_program(find(name + ".pte")).load_method("forward")
    p = [float(m.execute([x])[0].reshape(-1)[0]) for x in X]
    if min(p) < 0 or max(p) > 1: p = [float(torch.sigmoid(torch.tensor(v))) for v in p]  # logits -> probability
    s = metrics(p, y, 0.5); summary[name] = {"t0.5": s}
    print(f"{name:16} {s['acc']:6.3f} {s['prec']:6.3f} {s['rec']:6.3f} {s['f1']:6.3f} {s['fa']*100:11.2f}% ({s['fp']}) {s['fn']:5}")
    with open(os.path.join(ROOT, "results", f"real_scores_{name}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["score", "label", "wrong", "category", "platform", "text"])
        for r, sc, lab in sorted(zip(rows, p, y), key=lambda t: -t[1]):
            w.writerow([f"{sc:.4f}", lab, int((sc >= 0.5) != lab), r.get("category", ""), r.get("platform", ""), r["text"]])
    if name == "sms_tiny_int8":   # the model the app uses: show cut-off options
        print("   cut-offs for sms_tiny_int8 (score >= threshold counts as scam):")
        for th in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            t = metrics(p, y, th); summary[name][f"t{th}"] = t
            print(f"     {th:.1f}: recall {t['rec']:.3f}  false alarms {t['fa']*100:5.2f}%  F1 {t['f1']:.3f}")
json.dump(summary, open(os.path.join(ROOT, "results", "real_eval_summary.json"), "w"), indent=1)
print("\nSaved results/real_eval_summary.json and results/real_scores_<model>.csv (wrong=1 marks mistakes)")
