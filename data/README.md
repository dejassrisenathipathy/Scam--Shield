# Data

No SMS data is stored in this repository.

- Public training data: UCI SMS Spam Collection and the Shade63 Indian scam SMS dataset. `ml/prepare_sms.py` downloads, cleans, de-duplicates and splits them (80/10/10).
- Real-world evaluation set: 463 real Indian SMS (228 scam, 235 genuine) collected from team members' and volunteers' phones. It is kept private because it contains personal messages, bank details and phone numbers. Our scripts expect it at `data/real_test_463.csv` (columns: text, label).
