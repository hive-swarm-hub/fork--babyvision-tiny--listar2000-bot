"""BabyVision solver — visual reasoning on early visual understanding tasks.

Takes a JSON task on stdin (question, image_path, ans_type, options), prints the answer on stdout.
Saves full LLM trajectory to eval_results/trajectories/<index>.json if EVAL_TRAJECTORY_DIR is set.
"""

import sys
import os
import json
import base64

from openai import OpenAI


def solve(question: str, image_path: str, ans_type: str, options: list) -> str:
    client = OpenAI()

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    if ans_type == "choice" and options:
        opts = "\n".join(f"{i+1}. {o}" for i, o in enumerate(options))
        prompt = f"{question}\n\nOptions:\n{opts}\n\nAnswer with ONLY the option number (1, 2, 3, or 4)."
    else:
        prompt = f"{question}\n\nGive ONLY the answer, nothing else."

    messages = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": prompt},
        ]},
    ]

    model = os.environ.get("SOLVER_MODEL", "gpt-5.4-mini")
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=64,
    )

    raw_output = response.choices[0].message.content.strip()

    # Save trajectory if requested
    traj_dir = os.environ.get("EVAL_TRAJECTORY_DIR")
    idx = os.environ.get("EVAL_INDEX")
    if traj_dir and idx is not None:
        os.makedirs(traj_dir, exist_ok=True)
        # For trajectory, save text prompt only (not the base64 image — too large)
        trajectory = {
            "index": int(idx),
            "model": model,
            "prompt": prompt,
            "question": question,
            "image_path": image_path,
            "ans_type": ans_type,
            "options": options,
            "raw_response": raw_output,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                "completion_tokens": response.usage.completion_tokens if response.usage else None,
            },
        }
        with open(os.path.join(traj_dir, f"{idx}.json"), "w") as f:
            json.dump(trajectory, f, indent=2)

    return raw_output


if __name__ == "__main__":
    data = json.loads(sys.stdin.read().strip())
    print(solve(data["question"], data["image_path"], data["ans_type"], data.get("options", [])))
