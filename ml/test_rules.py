"""Check the Python rules copy (ml/risk_rules.py) against the 30 cases in the app's RulesTest.kt.

Same logic as RulesTest.kt: a rule fires if its pattern matches the raw text and its `unless`
pattern (if any) does not. A case passes if every mustFire rule fires and no mustNotFire rule fires.

Run from ~/scam_shield with .venv active:   python -m ml.test_rules
"""
import re
import sys

import ml.risk_rules as rr

CASES = [
    ("Dear customer your SBI KYC will expire today. Update now: sbi-kyc.xyz", ["kyc"], []),
    ("Your PAN card is blocked. Link Aadhaar immediately", ["kyc", "urgency"], []),
    ("Your account will be suspended within 24 hrs", ["account_block", "urgency"], []),
    ("Share the OTP sent to your phone with our agent to receive cashback", ["otp_ask"], []),
    ("123456 is your OTP. Do not share it with anyone. -HDFC Bank", [], ["otp_ask"]),
    ("Never share your PIN or OTP with anyone. SBI never asks for it.", [], ["otp_ask"]),
    ("Enter your UPI PIN to receive Rs 5000 cashback", ["upi_collect"], []),
    ("Your electricity power will be disconnected tonight at 9.30pm. Call 9876543210", ["electricity", "call_number"], []),
    ("Your EB bill payment of Rs 1,240 received. Thank you. -TNEB", [], ["electricity"]),
    ("Congratulations! You won Rs 25 lakh in KBC lucky draw", ["prize"], []),
    ("I won't be able to come today, sorry", [], ["prize", "urgency"]),
    ("Earn Rs 3000 daily by liking YouTube videos. Part time work from home", ["job"], []),
    ("Your FedEx parcel contains drugs and is seized by customs", ["courier"], []),
    ("Your Amazon package will be delivered today", [], ["courier", "urgency"]),
    ("This is Mumbai Police. A warrant is issued in your name. Digital arrest", ["police"], []),
    ("Pre-approved loan of Rs 5 lakh. Pay processing fee of Rs 999", ["loan", "fee"], []),
    ("Your loan EMI of Rs 4,500 is due on 05-Dec. -HDFC Bank", [], ["loan"]),
    ("Install AnyDesk so our executive can help you", ["remote_app"], []),
    ("Download update.apk to keep your account active", ["remote_app"], []),
    ("Pay registration fee of Rs 500 to confirm your job", ["fee"], []),
    ("Join our stock tips WhatsApp group for guaranteed returns", ["invest"], []),
    ("Earn 5% daily returns on crypto investment", ["invest"], []),
    ("Your e-challan of Rs 2000 is pending. Pay now at echallan-pay.top", ["challan"], []),
    ("Your FASTag will be deactivated, recharge here", ["challan"], []),
    ("Your HDFC credit card reward points worth Rs 7,850 expire today. Redeem now", ["reward_points"], []),
    ("Income tax refund of Rs 15,490 approved. Verify your account", ["tax_refund"], []),
    ("Meeting moved to 3pm today, see you there", [], ["urgency", "prize", "police"]),
    ("Your order has been shipped and will arrive in 2 days", [], ["courier", "urgency"]),
    ("Rs 500 credited to your a/c XX1234 by UPI. Avl bal Rs 10,234", [], ["upi_collect", "account_block"]),
    ("Last date to pay your college fee is 20 Nov", ["urgency"], ["fee"]),
]


def field(rule, name):
    if isinstance(rule, dict):
        return rule.get(name)
    if isinstance(rule, (tuple, list)) and not hasattr(rule, "_fields"):
        return None
    return getattr(rule, name, None)


def as_regex(p):
    if p is None or hasattr(p, "search"):
        return p
    return re.compile(str(p), re.IGNORECASE)


def find_rules():
    # risk_rules.RULES holds tuples: (id, category, pattern, weight, why, unless)
    if isinstance(getattr(rr, "RULES", None), list) and all(isinstance(r, tuple) and len(r) == 6 for r in rr.RULES):
        return "RULES", [(r[0], as_regex(r[2]), as_regex(r[5])) for r in rr.RULES]
    for name in dir(rr):
        obj = getattr(rr, name)
        if isinstance(obj, (list, tuple)) and obj and all(field(r, "id") and field(r, "pattern") for r in obj):
            return name, [(field(r, "id"), as_regex(field(r, "pattern")), as_regex(field(r, "unless"))) for r in obj]
    return None, None


def main():
    name, rules = find_rules()
    if rules is None:
        print("Could not find a rule list (items with 'id' and 'pattern') in ml/risk_rules.py.")
        print("Names in the module:", [n for n in dir(rr) if not n.startswith("_")])
        print("Send this output and ml/risk_rules.py to Claude.")
        sys.exit(2)
    ids = {r[0] for r in rules}
    print(f"Using {name}: {len(rules)} rules")

    def hits(text):
        return [rid for rid, pat, unless in rules
                if pat.search(text) and not (unless is not None and unless.search(text))]

    missing = sorted({r for _, f, nf in CASES for r in f + nf} - ids)
    if missing:
        print("Rule ids used in RulesTest.kt but missing from the Python copy:", missing)

    failures = 0
    for n, (text, must, must_not) in enumerate(CASES, 1):
        h = hits(text)
        ok = all(r in h for r in must) and not any(r in h for r in must_not)
        if not ok:
            failures += 1
            print(f"FAIL case {n}: \"{text}\"\n    fired {h}, expected {must}, not {must_not}")
    print(f"\n{len(CASES) - failures} of {len(CASES)} cases pass")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
