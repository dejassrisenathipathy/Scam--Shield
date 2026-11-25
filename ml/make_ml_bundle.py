import hashlib, os, zipfile, datetime

ROOT = os.path.expanduser("~/scam_shield")
WANTED = {
    "models": ["sms_tiny_fp32.pte", "sms_tiny_int8.pte", "sms_deep_fp32.pte", "sms_deep_int8.pte"],
    "support": ["vocab.json", "test_vectors.json", "models.md"],
    "test_data": ["test_1000.json"],
    "results": ["baseline_sms.json"],
}
SKIP = {"raw", ".git", "__pycache__", ".ipynb_checkpoints"}

found = {}
for d, dirs, files in os.walk(ROOT):
    dirs[:] = [x for x in dirs if x not in SKIP and not x.startswith(".")]
    for f in files:
        found.setdefault(f, []).append(os.path.join(d, f))

missing, dupes, plan = [], [], []
for folder, names in WANTED.items():
    for n in names:
        paths = found.get(n, [])
        if not paths:
            missing.append(n)
            continue
        if len(paths) > 1:
            paths.sort(key=os.path.getmtime, reverse=True)
            dupes.append((n, paths))
        plan.append((folder, n, paths[0]))

for n, paths in dupes:
    print(f"NOTE: {n} found {len(paths)} times, using the newest: {paths[0]}")
if missing:
    print("MISSING (not added):", ", ".join(missing))

today = datetime.date.today().isoformat()
out = os.path.join(ROOT, f"ml_handover_{today}.zip")
lines = [f"Scam Shield ML handover, {today}",
         "Exported with ExecuTorch 1.6.0 (XNNPACK); loads in the app's ExecuTorch Android 1.5.1.",
         "Input: int64 [1, 200] character ids (see vocab.json). Output: scam probability [1].", "",
         f"{'file':45} {'KB':>8}  sha256 (first 16)"]
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for folder, n, p in plan:
        z.write(p, f"{folder}/{n}")
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
        lines.append(f"{folder + '/' + n:45} {os.path.getsize(p)/1024:8.1f}  {h}")
    z.writestr("MANIFEST.txt", "\n".join(lines) + "\n")

print("\n".join(lines[4:]))
print(f"\nWrote {out} ({os.path.getsize(out)/1024:.1f} KB, {len(plan)} files + MANIFEST.txt)")
