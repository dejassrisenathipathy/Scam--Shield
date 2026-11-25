#!/usr/bin/env bash
set -e
adb shell mkdir -p /data/local/tmp/llama
adb push llama32_1b_8da4w.pte /data/local/tmp/llama/
adb push tokenizer.model /data/local/tmp/llama/
adb shell ls -l /data/local/tmp/llama
