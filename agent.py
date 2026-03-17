"""BabyVision solver — visual reasoning on early visual understanding tasks.

Takes a JSON task on stdin (question, image_path, ans_type, options), prints the answer on stdout.
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

    response = client.chat.completions.create(
        model=os.environ.get("SOLVER_MODEL", "gpt-4.1-mini"),
        messages=[
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": prompt},
            ]},
        ],
        temperature=0,
        max_tokens=64,
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    data = json.loads(sys.stdin.read().strip())
    print(solve(data["question"], data["image_path"], data["ans_type"], data.get("options", [])))
