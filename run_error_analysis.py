#!/usr/bin/env python3
"""
Error analysis for HW3: Find examples where models are correct on original
but fail on perturbed text. Run on a small sample for quick results.

Prerequisite: Run hw3.py first to create data.jsonl and perturbed_cache_N.json.

Usage:
  # Full run (all models, all examples in data)
  ./venv/bin/python run_error_analysis.py --data_path data.jsonl --output_dir .

  # Quick test (1 model, 50 examples) - ~2 min
  ./venv/bin/python run_error_analysis.py --data_path data.jsonl --max_examples 50 --models 1

  # For 15k run: use data from main run, limit to 500 examples for faster error analysis
  ./venv/bin/python run_error_analysis.py --data_path data.jsonl --max_examples 500 --output_dir .

Output: error_analysis.json (full detail), error_analysis.csv (for report tables)
"""

import json
import argparse
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

# Import from main hw3
from hw3 import (
    load_data_with_perturbations,
    get_nli_pipeline,
    _label_to_id,
    PERTURBATIONS,
)

MODELS = [
    'textattack/bert-base-uncased-MNLI',
    'textattack/roberta-base-MNLI',
    'microsoft/deberta-base-mnli',
    'typeform/distilbert-base-uncased-mnli',
]


def predict_one(pipe, premise: str, hypothesis: str) -> int:
    """Return predicted label id for a single premise-hypothesis pair."""
    try:
        result = pipe([premise, hypothesis])
        return _label_to_id(result[0]['label'])
    except Exception:
        return -1  # Unknown


def run_error_analysis(
    examples: list,
    perturbed_examples: dict,
    models: list,
    max_examples: Optional[int] = None,
) -> list[dict]:
    """
    Find failures: correct on original, wrong on perturbed.
    Returns list of dicts with model, perturbation_type, premise, hypothesis, etc.
    """
    if max_examples:
        examples = examples[:max_examples]
        perturbed_examples = {
            pname: plist[:max_examples]
            for pname, plist in perturbed_examples.items()
        }

    failures = []
    n = len(examples)

    for model_name in models:
        print(f"Loading {model_name}...")
        pipe = get_nli_pipeline(model_name)

        for i in tqdm(range(n), desc=f"Error analysis {model_name}", leave=False):
            ex = examples[i]
            pred_orig = predict_one(pipe, ex['premise'], ex['hypothesis'])
            correct_orig = pred_orig == ex['label']

            if not correct_orig:
                continue  # Only care about "correct on original"

            for pname, plist in perturbed_examples.items():
                pert = plist[i]
                pred_pert = predict_one(pipe, pert['premise'], pert['hypothesis'])
                correct_pert = pred_pert == ex['label']

                if not correct_pert:
                    failures.append({
                        'model': model_name,
                        'perturbation_type': pname,
                        'example_idx': i,
                        'premise': ex['premise'],
                        'hypothesis': ex['hypothesis'],
                        'perturbed_premise': pert['premise'],
                        'perturbed_hypothesis': pert['hypothesis'],
                        'label': ex['label'],
                        'label_name': ['entailment', 'neutral', 'contradiction'][ex['label']],
                        'split': ex['split'],
                        'pred_original': pred_orig,
                        'pred_perturbed': pred_pert,
                    })

    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to data.jsonl')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory')
    parser.add_argument('--max_examples', type=int, default=None, help='Limit examples for quick run (e.g., 50)')
    parser.add_argument('--models', type=int, default=None, help='Limit to first N models (e.g., 1 for quick test)')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    print(f"Loading data from {data_path}...")
    examples, perturbed_examples = load_data_with_perturbations(str(data_path))

    if perturbed_examples is None:
        cache_file = out_dir / f"perturbed_cache_{len(examples)}.json"
        if not cache_file.exists():
            raise FileNotFoundError(
                f"Perturbed cache not found: {cache_file}\n"
                "Run hw3.py first to generate perturbations, or use data.jsonl with full format."
            )
        print(f"Loading perturbations from {cache_file}...")
        with open(cache_file) as f:
            cache = json.load(f)
        perturbed_examples = {pname: cache['perturbed'][pname] for pname in PERTURBATIONS}

    models = MODELS
    if args.models:
        models = MODELS[: args.models]
        print(f"Using first {args.models} model(s) for quick test")

    if args.max_examples:
        print(f"Limiting to {args.max_examples} examples for quick run")

    failures = run_error_analysis(
        examples,
        perturbed_examples,
        models=models,
        max_examples=args.max_examples,
    )

    # Save JSON (full detail)
    out_json = out_dir / 'error_analysis.json'
    with open(out_json, 'w') as f:
        json.dump(failures, f, indent=2)
    print(f"Saved {len(failures)} failure examples to {out_json}")

    # Save CSV (for report tables)
    if failures:
        import pandas as pd
        df = pd.DataFrame(failures)
        out_csv = out_dir / 'error_analysis.csv'
        df.to_csv(out_csv, index=False)
        print(f"Saved {out_csv}")

        # Summary by model, perturbation, split
        summary = df.groupby(['model', 'perturbation_type', 'split']).size().reset_index(name='count')
        print("\nFailure counts by model / perturbation / split:")
        print(summary.to_string(index=False))
    else:
        print("No failure examples found (models were robust on this sample).")


if __name__ == '__main__':
    main()
