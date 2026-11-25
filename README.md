# SMS Scam Shield

On-device SMS scam and phishing detection for Indian users. It runs fully offline on the phone's Arm CPU with ExecuTorch, and the app has no INTERNET permission.

Bharat AI-SoC Challenge 2026 (Arm) · Cybersecurity & Digital Safety: on-device threat, phishing and scam detection · Team from PSG iTech: Deju (ML), Samrithi (SoC), Ashica (Android)

- Demo video: (https://youtu.be/F1M7qwkSGqo?si=WgIHjS3xDFWYkBnt)
- Technical report: (https://1drv.ms/w/c/c66c71a0fbcd3e5d/IQChMQ3C4Y-BRYnG___411_CAbkpBRqSZ6eJF_OnjZCiQgU?e=pGNope)

## How it works

1. An SMS arrives. The app joins multi-part messages and checks them on a background thread. Users can also share or paste text (for example a WhatsApp forward).
2. 18 India-specific rules check the sender type (business ID, shortcode or 10-digit mobile), links (official `.bank.in` / `.gov.in` sites, shorteners, IP links, an offline phishing blocklist) and scam language (KYC, e-challan, electricity cut-off, job, prize, investment, tax refund and more).
3. An int8 character-level CNN scores the text through ExecuTorch with the XNNPACK backend on the Arm CPU.
4. Risk fusion: risk = 1 - (1 - model score) x (1 - rules score). SUSPICIOUS at 0.39 or more, SCAM at 0.92 or more.
5. Only for flagged messages, Llama 3.2 1B Instruct (8da4w, through ExecuTorch) writes a 3-sentence explanation from the detected reasons. The detector decides; Llama only explains. If Llama takes over 60 s, a template explanation is shown.
6. The user sees a pop-up, the reasons, what to do next, one-tap reporting to 1930 and cybercrime.gov.in, and "Mark as safe". Nothing is deleted or blocked automatically.

## Results

| Measure | Result |
| --- | --- |
| Accuracy on 463 real Indian SMS, first model (public data only) | 57-62% |
| Accuracy on 463 real Indian SMS, final system (deep int8 + rules) | 90.9% (grouped 5-fold cross-validation, cut-offs chosen out-of-fold) |
| Detector latency, Galaxy S24 FE (Exynos 2400e) | tiny int8 0.048 ms (prime core), deep int8 0.31 ms (prime) / 1.57 ms (LITTLE) |
| Detector latency, Galaxy A71 (Snapdragon 730) | tiny int8 0.13 ms (big) / 0.25 ms (LITTLE) |
| Detector memory | about 5-6 MiB |
| Llama 3.2 1B, S24 FE, 4 fast-core threads | 0.8 s to first word, 47 tokens/s, 2.7 s per explanation, 903 MiB |
| Llama 3.2 1B, A71, 2 big-core threads | 3.1 s to first word, 14.1 tokens/s, about 9.4 s per explanation |
| Battery, Llama only on flagged (Run A) vs on every message (Run B) | [ ] mAh per message vs [ ] mAh per message |
| Model sizes | tiny int8 31.8 KB, deep int8 187.3 KB, Llama 821 MB |

The 90.9% is measured honestly: no message used to test a model was used to train it.

## Repository layout

    ml/        data prep, training, cross-validation, cut-offs, export to .pte
    models/    SMS detector .pte files, vocab.json, test_vectors.json, models.md
    url/       offline phishing blocklist and trusted-site allowlist
    android/   Android Studio project (Kotlin, Jetpack Compose)
    soc/       benchmark commands and results, Llama setup and prompt
    results/   metrics and plots
    docs/      technical report, diagrams, screenshots
    data/      how to get the datasets (no SMS stored here)

## Install and try the app

1. On an arm64 Android phone (Android 8 or newer), download the APK from Releases and install it (allow "Install unknown apps").
2. Open the app and allow SMS and notification permissions.
3. Paste a test message, for example: "Dear customer your SBI KYC expired. Update now: sbi-kyc-update.in"

## Run Llama on the phone

The Llama file is not in this repo (821 MB, and Meta's licence must be accepted by each user).

1. Accept the Llama 3.2 licence and download Llama-3.2-1B-Instruct from Meta or Hugging Face.
2. Export it with ExecuTorch 1.5.1 using the command in `soc/llama_export_command.txt` (8da4w quantisation, 4-bit embeddings, KV cache, max sequence length 512, XNNPACK).
3. Connect the phone with USB debugging on and run `soc/push_llama.sh` from the folder holding `llama32_1b_8da4w.pte` and `tokenizer.model`.
4. Restart the app and wait for "Llama ready".

## Build the app from source

1. Open `android/` in Android Studio and let Gradle sync.
2. Build, Build APK(s). The app uses `org.pytorch:executorch-android:1.5.1`.

## Reproduce the ML results

Python [3.x] in WSL Ubuntu:

    ./setup/bootstrap.sh
    source .venv/bin/activate
    python -m ml.prepare_sms      # public data: merge, clean, split
    python -m ml.baseline_sms     # TF-IDF baseline
    python -m ml.train_sms        # tiny and deep char-CNN on public data
    python -m ml.export_sms       # export to .pte (int8 + fp32), parity check
    python -m ml.cv_real          # needs the private real set: 5-fold CV
    python -m ml.cutoffs_real     # cut-offs chosen out-of-fold
    python -m ml.train_final      # final models shipped in the app

`models/test_vectors.json` holds inputs with expected scores, used to check the app gives the same output as Python.

## Reproduce the benchmarks

See `soc/benchmark_commands.md`. In short: build ExecuTorch's `executor_runner` and `llama_main` for arm64-v8a with the Android NDK, push them with the models to `/data/local/tmp/` with `adb`, and pin runs to LITTLE or big cores with `taskset`.

## Privacy

- No INTERNET permission (checked with Android Studio's APK Analyzer). Permissions: RECEIVE_SMS and POST_NOTIFICATIONS.
- Messages, verdicts and trusted senders stay in app-private storage on the phone.
- Llama sees only the reasons and the first 300 characters, and is told never to repeat links or numbers.

## Limitations

- SMS only; other chat apps work through Share.
- English first; Hindi, Tamil and mixed-script scams are only partly covered.
- The real test set has 463 messages, so the 90.9% has real uncertainty.
- Llama files are side-loaded for this prototype.

## Credits and licence

Code: MIT licence (see LICENSE). Built with Llama: Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright (c) Meta Platforms, Inc. Datasets: UCI SMS Spam Collection; Shade63 Indian scam SMS dataset. ExecuTorch and PyTorch by Meta.
