#!/usr/bin/env bash
set -euo pipefail

DATA="data/test.jsonl"
if [ ! -f "$DATA" ]; then
    echo "ERROR: $DATA not found. Run: bash prepare.sh" >&2
    exit 1
fi

TOTAL=$(wc -l < "$DATA")
CORRECT=0

echo "Evaluating $TOTAL problems..." >&2

while IFS= read -r line; do
    expected=$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin)['answer'])")
    got=$(echo "$line" | python3 agent.py 2>/dev/null || echo "ERROR")

    # Normalize: strip, lowercase
    got_norm=$(echo "$got" | xargs | tr '[:upper:]' '[:lower:]')
    exp_norm=$(echo "$expected" | xargs | tr '[:upper:]' '[:lower:]')

    if [ "$got_norm" = "$exp_norm" ]; then
        CORRECT=$((CORRECT + 1))
    fi
done < "$DATA"

ACCURACY=$(python3 -c "print(f'{$CORRECT / $TOTAL:.6f}')")

echo "---"
echo "accuracy:         $ACCURACY"
echo "correct:          $CORRECT"
echo "total:            $TOTAL"
