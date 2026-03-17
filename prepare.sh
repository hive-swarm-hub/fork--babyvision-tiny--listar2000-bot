#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/images
echo "Downloading BabyVision (tiny: 30 problems, seed=42)..."
python3 << 'PY'
from datasets import load_dataset
import json, pathlib, random

random.seed(42)
ds = load_dataset('UnipatAI/BabyVision', split='train')

indices = random.sample(range(len(ds)), 30)

out = pathlib.Path('data/test.jsonl')
with out.open('w') as f:
    for j, i in enumerate(sorted(indices)):
        row = ds[i]
        img_path = f'data/images/{j:04d}.jpg'
        img = row['image'].convert('RGB') if row['image'].mode != 'RGB' else row['image']
        img.save(img_path)

        answer = str(row['choiceAns']) if row['ansType'] == 'choice' else str(row['blankAns'] or '')

        f.write(json.dumps({
            'question': row['question'],
            'ans_type': row['ansType'],
            'options': row['options'] if row['options'] else [],
            'answer': answer,
            'image_path': img_path,
            'subtype': row['subtype'],
        }) + '\n')

print(f'Wrote 30 problems to {out}')
PY
echo "Done. $(wc -l < data/test.jsonl) problems in data/test.jsonl"
