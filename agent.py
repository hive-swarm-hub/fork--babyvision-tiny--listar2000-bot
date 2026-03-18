"""BabyVision solver — visual reasoning on early visual understanding tasks.

Takes a JSON task on stdin (question, image_path, ans_type, options), prints the answer on stdout.
Saves full LLM trajectory to eval_results/trajectories/<index>.json if EVAL_TRAJECTORY_DIR is set.
"""

import sys
import os
import json
import base64
import re
import io
from collections import Counter

from openai import OpenAI
from PIL import Image


def load_image_b64(image_path: str, min_size: int = 768) -> str:
    """Load image, upscale if too small, return base64."""
    img = Image.open(image_path)
    w, h = img.size
    if max(w, h) < min_size:
        scale = min_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def extract_answer(raw_output, ans_type):
    """Extract and clean the answer from model output."""
    lines = [l.strip() for l in raw_output.split("\n") if l.strip()]
    answer = lines[-1] if lines else raw_output
    answer = re.sub(r'\s*,\s*', ',', answer)
    answer = answer.rstrip('.')
    if ans_type == "choice":
        m = re.search(r'\b([0-4])\b', answer)
        if m:
            answer = m.group(1)
    return answer


def call_model(client, model, messages, temperature=0, max_tokens=4096):
    """Make an API call with retry on empty response."""
    response = client.chat.completions.create(
        model=model, messages=messages,
        temperature=temperature, max_completion_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    if not content or not content.strip():
        response = client.chat.completions.create(
            model=model, messages=messages,
            temperature=0.2, max_completion_tokens=max_tokens,
        )
        content = response.choices[0].message.content
    return content.strip() if content else ""


DESC_PROMPTS = [
    "Describe this image in detail. Focus on: the layout/grid structure, all visual elements (shapes, colors, patterns, numbers, letters), positions of elements, any differences or similarities between elements, and any spatial relationships. Be thorough and precise.",
    "Analyze this image carefully. List every distinct element you can see, its position, color, and shape. Note the overall structure (grid, rows, columns) and any patterns or anomalies.",
    "Look at this image and describe it systematically. Start from the top-left and work across and down. Note every shape, color, number, letter, and spatial relationship you observe.",
]


def single_run(client, model, hi_url, question, ans_type, options, desc_prompt, answer_temp):
    """One full describe-then-answer pipeline."""
    description = call_model(
        client, model,
        [{"role": "user", "content": [hi_url, {"type": "text", "text": desc_prompt}]}],
        temperature=0, max_tokens=2048,
    )
    if not description:
        description = "(no description available)"

    if ans_type == "choice" and options:
        opts = "\n".join(f"{i}. {o}" for i, o in enumerate(options))
        prompt = f"""Here is a detailed description of the image:
{description}

Now answer this question about the image:
{question}

Options:
{opts}

Think step by step, then give your final answer as ONLY the option number (0, 1, 2, or 3) on the last line."""
    else:
        prompt = f"""Question: {question}

Image analysis notes:
{description}

Look at the image carefully. Think step by step. Pay close attention to the exact answer format requested in the question. Put ONLY the final answer value on the last line."""

    raw = call_model(
        client, model,
        [{"role": "user", "content": [hi_url, {"type": "text", "text": prompt}]}],
        temperature=answer_temp, max_tokens=4096,
    )
    return raw, description


def solve(question: str, image_path: str, ans_type: str, options: list) -> str:
    client = OpenAI()

    img_b64 = load_image_b64(image_path)
    model = os.environ.get("SOLVER_MODEL", "gpt-5.4-mini")

    hi_url = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "high"}}

    # Run 3 independent describe-then-answer pipelines with different description prompts
    answers = []
    raw_outputs = []
    descriptions = []
    for i in range(3):
        raw, desc = single_run(
            client, model, hi_url, question, ans_type, options,
            DESC_PROMPTS[i], answer_temp=[0, 0.2, 0.4][i],
        )
        raw_outputs.append(raw)
        descriptions.append(desc)
        answers.append(extract_answer(raw, ans_type))

    # Majority vote
    counter = Counter(answers)
    answer = counter.most_common(1)[0][0]
    best_idx = answers.index(answer)

    # Save trajectory
    traj_dir = os.environ.get("EVAL_TRAJECTORY_DIR")
    idx = os.environ.get("EVAL_INDEX")
    if traj_dir and idx is not None:
        os.makedirs(traj_dir, exist_ok=True)
        trajectory = {
            "index": int(idx),
            "model": model,
            "question": question,
            "image_path": image_path,
            "ans_type": ans_type,
            "options": options,
            "all_answers": answers,
            "voted_answer": answer,
            "description": descriptions[best_idx],
            "raw_response": raw_outputs[best_idx],
            "parsed_answer": answer,
        }
        with open(os.path.join(traj_dir, f"{idx}.json"), "w") as f:
            json.dump(trajectory, f, indent=2)

    return answer


if __name__ == "__main__":
    data = json.loads(sys.stdin.read().strip())
    print(solve(data["question"], data["image_path"], data["ans_type"], data.get("options", [])))
