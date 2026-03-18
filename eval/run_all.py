"""Evaluate agent.py on BabyVision. Concurrent."""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

data_path = sys.argv[1]
max_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8

with open(data_path) as f:
    tasks = [json.loads(line) for line in f]

total = len(tasks)
correct = 0
completed = 0


def eval_one(task):
    try:
        proc = subprocess.run(
            ["python3", "agent.py"],
            input=json.dumps(task), capture_output=True, text=True, timeout=30,
        )
        got = proc.stdout.strip().lower()
        exp = task["answer"].lower()
        return got == exp
    except Exception:
        return False


print(f"Evaluating {total} problems ({max_workers} concurrent)...", file=sys.stderr)

with ThreadPoolExecutor(max_workers=max_workers) as pool:
    futures = {pool.submit(eval_one, t): t for t in tasks}
    for future in as_completed(futures):
        completed += 1
        if future.result():
            correct += 1
        print(f"  {completed}/{total} done, {correct} correct", file=sys.stderr, end="\r")

print(file=sys.stderr)
print("---")
print(f"accuracy:         {correct / total:.6f}" if total > 0 else "accuracy:         0.000000")
print(f"correct:          {correct}")
print(f"total:            {total}")
