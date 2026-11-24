#!/usr/bin/env bash
# One-shot cron wrapper for run_gen_colon_1p3b.sh (scheduled 2026-07-31 00:05):
# sends a Telegram notification at start (+ a summary when done), picks the 3
# freest GPUs, and removes its own crontab entry so it never re-fires.
# Telegram creds are read from the MCP config at runtime — no secrets here.
set -u
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
ROOT=/mnt/ssd2/hyun2/AUNet
OUT=$ROOT/reports/niah/gen_colon
mkdir -p "$OUT"
exec >>"$OUT/scheduled_run.log" 2>&1
echo "=== scheduled run fired: $(date) ==="

# self-remove the crontab entry (one-shot; would otherwise re-fire yearly)
crontab -l 2>/dev/null | grep -vF "run_gen_colon_scheduled.sh" | crontab -

TOK=$(jq -r '.mcpServers.telegram.env.TELEGRAM_BOT_TOKEN' "$HOME/.claude.json")
CHAT=$(jq -r '.mcpServers.telegram.env.TELEGRAM_CHAT_ID' "$HOME/.claude.json")
tg(){ curl -s -m 20 "https://api.telegram.org/bot$TOK/sendMessage" \
        -d chat_id="$CHAT" --data-urlencode text="$1" >/dev/null; }

# 3 freest GPUs by memory used
GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
       | sort -t, -k2 -n | head -3 | cut -d, -f1 | paste -sd' ')

tg "[AUNet] S-NIAH-1/2/3 generation-style eval (colon vs no-colon needle, subword Llama 1.3B) STARTING on $(hostname), GPUs: ${GPUS}. Results -> reports/niah/gen_colon/"

rm -f "$OUT"/gen_S*.jsonl "$OUT"/per_S*.jsonl
GPUS="$GPUS" bash "$ROOT/scripts/niah/run_gen_colon_1p3b.sh"

SUM=$(jq -rs 'group_by(.task + "|" + .cond)[]
              | "S-NIAH-\(.[0].task) \(.[0].cond): \(map(.gen_match)|add/length*100|round)%"' \
      "$OUT"/gen_S*.jsonl 2>/dev/null)
tg "[AUNet] S-NIAH gen-style colon eval DONE (mean gen_match over lengths/depths):
${SUM:-no results — check reports/niah/gen_colon/scheduled_run.log}"
echo "=== scheduled run finished: $(date) ==="
