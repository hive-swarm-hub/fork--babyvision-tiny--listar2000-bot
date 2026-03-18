"""Evaluate agent.py on BabyVision. Concurrent. Saves trajectories."""
import json
import subprocess
import sys
import os
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

data_path = sys.argv[1]
max_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8

with open(data_path) as f:
    tasks = [json.loads(line) for line in f]

total = len(tasks)
correct = 0
completed = 0
results = []


def eval_one(idx, task):
    try:
        proc = subprocess.run(
            ["python3", "agent.py"],
            input=json.dumps(task), capture_output=True, text=True, timeout=30,
        )
        got = proc.stdout.strip()
        exp = task["answer"]
        passed = got.lower() == exp.lower()

        return {
            "index": idx,
            "passed": passed,
            "predicted": got,
            "expected": exp,
            "question": task["question"][:200],
            "ans_type": task["ans_type"],
            "subtype": task.get("subtype", ""),
            "agent_stderr": proc.stderr[:500] if proc.stderr else None,
            "exit_code": proc.returncode,
        }
    except Exception as e:
        return {
            "index": idx,
            "passed": False,
            "predicted": None,
            "expected": task["answer"],
            "error": str(e),
        }


print(f"Evaluating {total} problems ({max_workers} concurrent)...", file=sys.stderr)

with ThreadPoolExecutor(max_workers=max_workers) as pool:
    futures = {pool.submit(eval_one, i, t): i for i, t in enumerate(tasks)}
    for future in as_completed(futures):
        completed += 1
        result = future.result()
        results.append(result)
        if result["passed"]:
            correct += 1
        print(f"  {completed}/{total} done, {correct} correct", file=sys.stderr, end="\r")

print(file=sys.stderr)

# Save trajectory
os.makedirs("eval_results", exist_ok=True)
timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
trajectory_path = f"eval_results/{timestamp}.jsonl"
results.sort(key=lambda r: r["index"])
with open(trajectory_path, "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")
print(f"Trajectory saved to {trajectory_path}", file=sys.stderr)

print("---")
print(f"accuracy:         {correct / total:.6f}" if total > 0 else "accuracy:         0.000000")
print(f"correct:          {correct}")
print(f"total:            {total}")
