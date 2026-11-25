import csv, io, os, urllib.request, zipfile
from urllib.parse import urlsplit

OUT = "url"
TOP_N = 10000

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "scam-shield-student-project"})
    return urllib.request.urlopen(req, timeout=60).read()

def host(u):
    u = u.strip()
    if not u or u.startswith("#"): return None
    if "://" not in u: u = "http://" + u
    h = (urlsplit(u).hostname or "").lower().removeprefix("www.")
    return h or None

# Allowlist: Tranco top sites + Indian official domains
tranco = zipfile.ZipFile(io.BytesIO(get("https://tranco-list.eu/top-1m.csv.zip")))
rows = csv.reader(io.TextIOWrapper(tranco.open(tranco.namelist()[0]), "utf-8"))
allow = {r[1].lower() for _, r in zip(range(TOP_N), rows)}
allow |= {"onlinesbi.sbi", "sbi.co.in", "sbicard.com", "hdfcbank.com", "icicibank.com", "axisbank.com",
          "kotak.com", "npci.org.in", "uidai.gov.in", "incometax.gov.in", "indiapost.gov.in",
          "echallan.parivahan.gov.in", "cybercrime.gov.in", "paytm.com", "phonepe.com", "amazon.in", "flipkart.com"}

# Blocklist: OpenPhish (phishing) + URLhaus (malware links)
block = set()
for name, url in [("OpenPhish", "https://openphish.com/feed.txt"),
                  ("Phishing.Army", "https://phishing.army/download/phishing_army_blocklist.txt")]:
    try:
        hosts = {h for h in map(host, get(url).decode("utf-8", "ignore").splitlines()) if h}
        print(f"{name}: {len(hosts)} hosts"); block |= hosts
    except Exception as e:
        print(f"{name}: download failed ({e}), skipped")

# Never block a big real site just because someone hosted one bad page on it
def on_allowed(h):
    p = h.split(".")
    return any(".".join(p[i:]) in allow for i in range(len(p) - 1))
removed = block & allow
block -= removed

os.makedirs(OUT, exist_ok=True)
open(f"{OUT}/url_allowlist.txt", "w").write("\n".join(sorted(allow)) + "\n")
open(f"{OUT}/url_blocklist.txt", "w").write("\n".join(sorted(block)) + "\n")
print(f"Allowlist: {len(allow)} domains -> {OUT}/url_allowlist.txt")
print(f"Blocklist: {len(block)} hosts (removed {len(removed)} on popular sites) -> {OUT}/url_blocklist.txt")
