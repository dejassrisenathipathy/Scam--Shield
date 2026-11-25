"""Python copy of the app's RiskEngine.assess() (RiskEngine.kt, rules tuned 15 Nov).

Scores a message on the laptop exactly the way the phone does, so cut-offs can be
chosen in Python. If RiskEngine.kt changes, change this file too.
Not copied: "Mark safe" trusted senders (needs the phone's saved list).
"""
import re

I = re.IGNORECASE


def _r(p: str) -> re.Pattern:
    return re.compile(p, I)


# (id, category, pattern, weight, reason, unless)
RULES = [
    ("kyc", "bank_kyc", _r(r"\b(kyc|pan ?card|pan|aadhaa?r)\b.{0,40}\b(updat|expir|block|suspend|link|verif)"), 0.20, "Asks you to update KYC, PAN or Aadhaar", None),
    ("account_block", "bank_kyc", _r(r"\b(account|a/c|card|sim|number)\b.{0,30}\b(block|suspend|deactivat|freez|clos)"), 0.15, "Threatens to block your account, card or SIM", None),
    ("urgency", "other", _r(r"\b(urgent(ly)?|immediately|asap|within \d+ ?(hrs?|hours?|mins?|minutes?)|in \d+ ?(hrs?|hours?|mins?|minutes?)|last (warning|chance|date)|act now|before midnight)\b"), 0.15, "Pressures you to act quickly", None),
    ("otp_ask", "upi_otp", _r(r"\b(share|send|tell|provide|forward)\b.{0,20}\b(otp|pin|cvv|password)\b"), 0.30, "Asks you to share an OTP, PIN or password", _r(r"\b(do not|don't|dont|never|not to)\b.{0,15}\b(share|send|tell|disclose)")),
    ("upi_collect", "upi_otp", _r(r"\b(enter|use)\b.{0,15}\b(upi )?pin\b.{0,40}\b(receive|get|claim|credit)"), 0.30, "Asks for your UPI PIN to 'receive' money (a PIN is only needed to pay)", None),
    ("electricity", "bill_electricity", _r(r"\b(electricity|power|light bill|eb bill|ebill|bijli)\b.{0,60}\b(disconnect|cut|discontinu)"), 0.30, "Threatens to cut your electricity", None),
    ("prize", "lottery_prize", _r(r"\b(won(?!'t)|winner|lottery|prize|lucky draw|jackpot|kbc)\b"), 0.20, "Offers a prize or lottery win", None),
    ("job", "job_task", _r(r"\b(part[- ]?time|work from home|daily (income|earning)|earn (rs\.?|inr|₹) ?\d|like (youtube )?videos|simple tasks?)\b"), 0.20, "Offers easy money or a part-time job", None),
    ("courier", "delivery_customs", _r(r"\b(parcel|courier|package|shipment|customs|fedex|dhl)\b.{0,60}\b(held|hold|seized|pending|fee|charge|illegal|drugs|narcotic)"), 0.25, "Claims a parcel is held, seized or illegal", None),
    ("police", "police_digital_arrest", _r(r"\b(police|cbi|ncb|narcotics|cyber cell|arrest|warrant|digital arrest)\b"), 0.25, "Claims to be police or a government agency", None),
    ("loan", "loan", _r(r"\b(pre-?approved|instant) loan\b|\bloan\b.{0,40}\b(approved|processing fee|without documents)"), 0.15, "Offers a loan that needs a fee", None),
    ("call_number", "other", _r(r"\b(call|contact|whatsapp|msg)\b.{0,25}(\+?91[- ]?)?[6-9]\d{9}\b"), 0.15, "Asks you to call or WhatsApp a personal number", None),
    ("remote_app", "other", _r(r"\b(anydesk|teamviewer|quicksupport|rustdesk)\b|\.apk\b"), 0.30, "Asks you to install a screen-sharing or unknown app", None),
    ("fee", "other", _r(r"\b(processing|registration|clearance|verification|delivery) (fee|charge)s?\b"), 0.20, "Asks you to pay a fee first", None),
    ("invest", "investment", _r(r"\b(stock tips?|trading (group|tips)|ipo allot|guaranteed (return|profit)s?|double your money|\d+ ?% (daily|weekly|monthly) (return|profit)s?|crypto)\b"), 0.25, "Promises guaranteed or very high investment returns", None),
    ("challan", "traffic_challan", _r(r"\b(e-?challan|traffic (fine|violation)|fastag)\b.{0,60}\b(pay|pending|block|due|deactivat|recharge)"), 0.20, "Claims you owe a traffic fine or FASTag payment", None),
    ("reward_points", "card_reward", _r(r"\breward points?\b.{0,40}\b(expir|redeem|lapse)"), 0.25, "Says your card reward points are expiring", None),
    ("tax_refund", "tax_refund", _r(r"\b(income tax|it) refund\b.{0,40}\b(approved|pending|claim|verify|link)"), 0.25, "Promises an income tax refund through a link", None),
]

LINK = re.compile(
    r"(https?://\S+|www\.\S+|\b[a-z0-9-]+(\.[a-z0-9-]+)*\.(in|com|net|org|xyz|top|info|live|online|site|club|co|me|io|ly|gl|link|click|app|shop|buzz)\b(/\S*)?)",
    I,
)
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "cutt.ly", "rb.gy", "shorturl.at", "tiny.cc", "ow.ly", "rebrand.ly"}
RISKY_ENDINGS = (".xyz", ".top", ".live", ".online", ".site", ".club", ".info", ".click", ".link", ".shop", ".buzz")
BRANDS = ("sbi", "hdfc", "icici", "axis", "kotak", "pnb", "canara", "paytm", "phonepe", "gpay", "upi", "npci", "rbi", "uidai", "aadhaar", "incometax", "epfo", "indiapost", "tneb", "tangedco", "bescom", "amazon", "flipkart")
OFFICIAL = ("onlinesbi.sbi", "sbi.co.in", "sbicard.com", "hdfcbank.com", "icicibank.com", "axisbank.com", "kotak.com", "paytm.com", "phonepe.com", "amazon.in", "amazon.com", "flipkart.com", "npci.org.in", "uidai.gov.in", "incometax.gov.in", "indiapost.gov.in")
IP = re.compile(r"\d{1,3}(\.\d{1,3}){3}")


def sender_type(sender: str) -> str:
    s = sender.replace(" ", "")
    if sender == "Pasted":
        return "unknown"
    if re.fullmatch(r"\+?\d{10,13}", s):
        return "mobile"
    if re.fullmatch(r"\d{3,6}", s):
        return "shortcode"
    if re.fullmatch(r"[A-Za-z]{2}-[A-Za-z0-9]{3,9}(-[A-Za-z])?", s):
        return "business"
    return "unknown"


def find_links(text: str) -> list[str]:
    return [m.group(0).rstrip(".,)") for m in LINK.finditer(text)]


def host_of(link: str) -> str:
    h = link.lower().removeprefix("https://").removeprefix("http://")
    for sep in "/?:":
        h = h.split(sep, 1)[0]
    return h.removeprefix("www.")


def is_official(host: str) -> bool:
    return (host.endswith((".bank.in", ".gov.in", ".nic.in"))
            or any(host == o or host.endswith("." + o) for o in OFFICIAL))


def assess(sender: str, text: str, sms_score: float) -> dict:
    """Same maths as RiskEngine.assess(). Returns risk, rule part and reasons."""
    reasons, cats, rule = [], [], 0.0
    for _id, cat, pat, w, why, unless in RULES:
        if pat.search(text) and not (unless and unless.search(text)):
            rule += w
            reasons.append(why)
            if cat != "other":
                cats.append((cat, w))

    kind = sender_type(sender)
    if kind == "mobile" and cats:
        rule += 0.15
        reasons.append("Claims to be an organisation but came from a personal mobile number")
    if kind == "business":
        rule = max(rule - 0.10, 0.0)

    for link in find_links(text):
        host = host_of(link)
        official = is_official(host)
        if host in SHORTENERS:
            rule += 0.20; reasons.append(f"Uses a short link ({host})")
        elif IP.fullmatch(host):
            rule += 0.25; reasons.append("Link uses a bare IP address")
        elif not official and any(b in host for b in BRANDS):
            rule += 0.35; reasons.append(f"Link {host} uses a brand name but is not official")
        elif not official and host.endswith(RISKY_ENDINGS):
            rule += 0.15; reasons.append(f"Link {host} uses a cheap web address")

    rule_part = min(max(rule, 0.0), 1.0)
    risk = 1.0 - (1.0 - sms_score) * (1.0 - rule_part)
    category = max(cats, key=lambda c: c[1])[0] if cats else "other"
    return {"risk": risk, "rule_part": rule_part, "sender_type": kind,
            "category": category, "reasons": list(dict.fromkeys(reasons))}


if __name__ == "__main__":
    # Quick sanity checks: python -m ml.risk_rules
    checks = [
        ("VM-TNEBLT", "Dear consumer your electricity power will be disconnected tonight. Call 9876543210 immediately", 0.1, True),
        ("9876543210", "Your SBI KYC expired, update now at sbi-kyc-update.in", 0.2, True),
        ("JD-HDFCBK", "Your OTP is 482913. Do not share it with anyone. -HDFC Bank", 0.05, False),
        ("AX-ECHLLN", "Challan paid. View at echallan.parivahan.gov.in", 0.05, False),
    ]
    ok = True
    for sender, text, s, scam in checks:
        out = assess(sender, text, s)
        good = (out["rule_part"] > 0.25) == scam
        ok &= good
        print(f"{'OK ' if good else 'BAD'} risk={out['risk']:.2f} rules={out['rule_part']:.2f} {out['reasons']}")
    print("All sanity checks passed" if ok else "Some checks FAILED")
