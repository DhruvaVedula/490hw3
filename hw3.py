#!/usr/bin/env python3
"""
HW3: Syntactic Complexity and Model Robustness
CS 490: Natural Language Processing · Spring 2026

Reproducible pipeline for:
- Part 1: Syntactic complexity metrics (CFG depth, subject-verb distance, etc.)
- Part 2: MultiNLI dataset loading, 4 model baselines
- Part 3: Meaning-preserving perturbations
- Part 4: Model evaluation on original + perturbed data
- Output: perf.csv, complex.csv
"""

import json
import random
import argparse
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm

# Reproducibility (PyTorch: https://pytorch.org/docs/stable/notes/randomness.html)
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# =============================================================================
# Part 1: Syntactic Complexity Metrics
# =============================================================================

def setup_parser():
    """Initialize Stanza pipeline for constituency and dependency parsing."""
    import stanza  # type: ignore
    # depparse requires lemma; constituency needs tokenize,pos
    processors = 'tokenize,pos,lemma,constituency,depparse'
    stanza.download('en', processors=processors, verbose=False)
    return stanza.Pipeline('en', processors=processors, verbose=False)


def cfg_tree_depth(doc) -> float:
    """
    Metric 1: Maximum depth of the CFG constituency parse tree.
    Deeper trees = more nested, complex structures.
    """
    depths = []
    for sent in doc.sentences:
        if hasattr(sent, 'constituency') and sent.constituency is not None:
            depth = _tree_depth(sent.constituency)
            depths.append(depth)
    return float(np.mean(depths)) if depths else 0.0


def _tree_depth(node) -> int:
    """Recursively compute max depth of a constituency tree node."""
    if not hasattr(node, 'children') or not node.children:
        return 1
    return 1 + max(_tree_depth(c) for c in node.children)


def subject_verb_distance(doc) -> float:
    """
    Metric 2: Average distance (in words) between subject and main verb.
    Longer dependencies = harder to process.
    """
    distances = []
    for sent in doc.sentences:
        if not hasattr(sent, 'words'):
            continue
        words = sent.words
        for i, w in enumerate(words):
            if w.deprel == 'nsubj' and w.head > 0:
                head_idx = w.head - 1  # 1-indexed in Stanza
                if head_idx < len(words):
                    dist = abs(i - head_idx)
                    distances.append(dist)
    return float(np.mean(distances)) if distances else 0.0


def mean_dependency_length(doc) -> float:
    """
    Metric 3: Mean dependency length (distance between head and dependent).
    Higher = more long-range dependencies = more complex.
    """
    lengths = []
    for sent in doc.sentences:
        if not hasattr(sent, 'words'):
            continue
        words = sent.words
        for i, w in enumerate(words):
            if w.head > 0:
                head_idx = w.head - 1
                if head_idx < len(words):
                    lengths.append(abs(i - head_idx))
    return float(np.mean(lengths)) if lengths else 0.0


def compute_complexity(text: str, nlp) -> dict:
    """Compute all 3 complexity metrics for a text."""
    try:
        doc = nlp(text)
        return {
            'cfg_depth': cfg_tree_depth(doc),
            'subj_verb_dist': subject_verb_distance(doc),
            'mean_dep_len': mean_dependency_length(doc),
        }
    except Exception:
        return {'cfg_depth': 0.0, 'subj_verb_dist': 0.0, 'mean_dep_len': 0.0}


# =============================================================================
# Part 2: Dataset
# =============================================================================

def load_multinli(sample_size: int = 15000):
    """
    Load MultiNLI dev sets, sample equally from matched/mismatched,
    preserve label distribution. Returns list of dicts with premise, hypothesis, label, split.
    """
    ds = load_dataset('nyu-mll/multi_nli', split=None)
    matched = ds['validation_matched']
    mismatched = ds['validation_mismatched']

    # Filter out -1 labels
    matched = matched.filter(lambda x: x['label'] != -1)
    mismatched = mismatched.filter(lambda x: x['label'] != -1)

    n_each = sample_size // 2
    matched_sampled = matched.shuffle(seed=RANDOM_SEED).select(range(min(n_each, len(matched))))
    mismatched_sampled = mismatched.shuffle(seed=RANDOM_SEED).select(range(min(n_each, len(mismatched))))

    examples = []
    for row in matched_sampled:
        ex = dict(row)
        examples.append({
            'premise': ex['premise'],
            'hypothesis': ex['hypothesis'],
            'label': ex['label'],
            'split': 'matched',
        })
    for row in mismatched_sampled:
        ex = dict(row)
        examples.append({
            'premise': ex['premise'],
            'hypothesis': ex['hypothesis'],
            'label': ex['label'],
            'split': 'mismatched',
        })

    random.shuffle(examples)
    return examples


def save_data_jsonl(examples, path: str = 'data.jsonl'):
    """Save dataset to data.jsonl."""
    with open(path, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')


def load_data_jsonl(path: str = 'data.jsonl'):
    """Load dataset from data.jsonl."""
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


# =============================================================================
# Part 3: Perturbations
# =============================================================================

def perturb_relative_clause(text: str, nlp, doc=None) -> str:
    """
    Perturbation 1: Convert "ADJ NOUN" to "NOUN, who is ADJ,".
    E.g. "The wealthy investor" -> "The investor, who is wealthy,"
    """
    try:
        if doc is None:
            doc = nlp(text)
        new_tokens = []
        i = 0
        while i < len(doc.sentences):
            sent = doc.sentences[i]
            if not hasattr(sent, 'words'):
                return text
            words = sent.words
            j = 0
            while j < len(words):
                w = words[j]
                # Look for amod (adjectival modifier) + noun
                if w.deprel == 'amod' and w.head > 0:
                    head_idx = w.head - 1
                    if head_idx < len(words) and head_idx > j:
                        # Insert "X, who is ADJ," before noun
                        adj_text = w.text
                        noun = words[head_idx].text
                        # Simple heuristic: prepend "The" if at start
                        prefix = ' '.join(ww.text for ww in words[:j])
                        suffix = ' '.join(ww.text for ww in words[head_idx + 1:])
                        return f"{prefix} {noun}, who is {adj_text}, {suffix}".strip()
                j += 1
            i += 1
        return text
    except Exception:
        return text


def perturb_appositive_simple(text: str, nlp, doc=None) -> str:
    """
    Perturbation 2: Add simple appositive after first NP.
    E.g. "The investor bought stock" -> "The investor, a key stakeholder, bought stock"
    """
    try:
        if doc is None:
            doc = nlp(text)
        for sent in doc.sentences:
            if not hasattr(sent, 'words'):
                continue
            words = sent.words
            for i, w in enumerate(words):
                if w.upos == 'NOUN' and i > 0 and words[i - 1].upos in ('DET', 'ADJ'):
                    # Insert appositive after this noun
                    noun_phrase_end = i
                    while noun_phrase_end < len(words) - 1 and words[noun_phrase_end + 1].deprel in ('amod', 'compound', 'det'):
                        noun_phrase_end += 1
                    before = ' '.join(ww.text for ww in words[:noun_phrase_end + 1])
                    after = ' '.join(ww.text for ww in words[noun_phrase_end + 1:])
                    appositive = "a notable participant"
                    return f"{before}, {appositive}, {after}".strip()
        return text
    except Exception:
        return text


def perturb_extra_relative(text: str, nlp, doc=None) -> str:
    """
    Perturbation 3: Add extra relative clause with irrelevant info.
    E.g. "The wealthy investor" -> "The wealthy investor, who is related to Sam's older sister,"
    """
    try:
        if doc is None:
            doc = nlp(text)
        for sent in doc.sentences:
            if not hasattr(sent, 'words'):
                continue
            words = sent.words
            for i, w in enumerate(words):
                if w.upos == 'NOUN' and i < len(words) - 1:
                    before = ' '.join(ww.text for ww in words[:i + 1])
                    after = ' '.join(ww.text for ww in words[i + 1:])
                    extra = "who is known to the local community"
                    return f"{before}, {extra}, {after}".strip()
        return text
    except Exception:
        return text


PERTURBATIONS = {
    'relative_clause': perturb_relative_clause,
    'appositive': perturb_appositive_simple,
    'extra_relative': perturb_extra_relative,
}


def _process_chunk(args):
    """Worker: process a chunk of examples. Returns (complex_rows, perturbed_dict)."""
    chunk, _ = args
    import stanza
    processors = 'tokenize,pos,lemma,constituency,depparse'
    nlp = stanza.Pipeline('en', processors=processors, verbose=False)
    complex_rows = []
    perturbed = {pname: [] for pname in PERTURBATIONS}
    for ex in chunk:
        prem, hyp = ex['premise'], ex['hypothesis']
        prem_doc = nlp(prem)
        hyp_doc = nlp(hyp)
        orig_text = f"{prem} {hyp}"
        orig_complex = compute_complexity(orig_text, nlp)
        for metric, val in orig_complex.items():
            complex_rows.append({'perturbation method': 'original', 'metric type': metric, 'value': val})
        for pname, pfunc in PERTURBATIONS.items():
            pert_prem = pfunc(prem, nlp, doc=prem_doc)
            pert_hyp = pfunc(hyp, nlp, doc=hyp_doc)
            pert_text = f"{pert_prem} {pert_hyp}"
            pert_complex = compute_complexity(pert_text, nlp)
            perturbed[pname].append({'premise': pert_prem, 'hypothesis': pert_hyp, 'text': pert_text, 'label': ex['label'], 'split': ex['split']})
            for metric, val in pert_complex.items():
                complex_rows.append({'perturbation method': pname, 'metric type': metric, 'value': val})
    return complex_rows, perturbed


# =============================================================================
# Part 4: Models
# =============================================================================

def get_nli_pipeline(model_name: str):
    """Get HuggingFace NLI pipeline for a model. Uses [premise, hypothesis] pair format."""
    from transformers.pipelines import pipeline
    return pipeline(
        'text-classification',
        model=model_name,
        device=0 if torch.cuda.is_available() else -1,
        truncation=True,
        max_length=512,
    )


def _label_to_id(label_str: str) -> int:
    """Map pipeline label string to MNLI label id (0=entailment, 1=neutral, 2=contradiction)."""
    s = label_str.lower()
    if 'entailment' in s or 'entail' in s:
        return 0
    if 'contradiction' in s or 'contradict' in s:
        return 2
    return 1


def evaluate_model(pipe, examples, perturbed_list=None, perturbation_name=None, batch_size: int = 32):
    """
    Evaluate model on examples. Returns accuracy.
    perturbed_list: optional list of dicts with 'premise', 'hypothesis' (same length as examples).
    """
    correct = 0
    pairs = []
    for i, ex in enumerate(examples):
        if perturbed_list is not None:
            prem, hyp = perturbed_list[i]['premise'], perturbed_list[i]['hypothesis']
        else:
            prem, hyp = ex['premise'], ex['hypothesis']
        pairs.append((prem, hyp))

    for start in tqdm(range(0, len(pairs), batch_size), desc=f"Eval {perturbation_name or 'baseline'}", leave=False):
        batch = pairs[start:start + batch_size]
        batch_labels = [ex['label'] for ex in examples[start:start + batch_size]]
        try:
            # NLI: pass list of (premise, hypothesis) pairs
            inputs = [[p[0], p[1]] for p in batch]
            results = pipe(inputs)
            for j, res in enumerate(results):
                pred_id = _label_to_id(res['label'])
                if pred_id == batch_labels[j]:
                    correct += 1
        except Exception:
            # Fallback: one at a time
            for j, (prem, hyp) in enumerate(batch):
                try:
                    result = pipe([prem, hyp])
                    pred_id = _label_to_id(result[0]['label'])
                    if pred_id == batch_labels[j]:
                        correct += 1
                except Exception:
                    pass
    return correct / len(examples) if examples else 0.0


# =============================================================================
# Main
# =============================================================================

def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.data_path and Path(args.data_path).exists():
        print(f"Loading data from {args.data_path}...")
        examples = load_data_jsonl(args.data_path)
    else:
        print("Loading MultiNLI...")
        examples = load_multinli(sample_size=args.sample_size)
        save_data_jsonl(examples, str(out_dir / 'data.jsonl'))
    print(f"Using {len(examples)} examples")

    # Check for cached perturbations (skip slow parsing when re-running with --data_path)
    cache_file = out_dir / f"perturbed_cache_{len(examples)}.json"
    if args.data_path and cache_file.exists():
        print(f"Loading cached perturbations from {cache_file}...")
        with open(cache_file) as f:
            cache = json.load(f)
        perturbed_examples = {pname: cache['perturbed'][pname] for pname in PERTURBATIONS}
        complex_agg = pd.DataFrame(cache['complex_rows']).groupby(
            ['perturbation method', 'metric type']
        ).agg({'value': 'mean'}).reset_index()
        complex_agg.to_csv(str(out_dir / 'complex.csv'), index=False)
        print("Loaded cache, skipping perturbations.")
    else:
        workers = getattr(args, 'workers', 1) or 1
        if workers > 1:
            import stanza
            print(f"Using {workers} workers for parallel perturbation...")
            stanza.download('en', processors='tokenize,pos,lemma,constituency,depparse', verbose=False)
            chunk_size = max(1, (len(examples) + workers - 1) // workers)
            chunks = [(examples[i:i + chunk_size], i) for i in range(0, len(examples), chunk_size)]
            with Pool(workers) as pool:
                results = list(tqdm(pool.imap(_process_chunk, chunks), total=len(chunks), desc="Perturbing"))
            complex_rows = []
            perturbed_examples = {pname: [] for pname in PERTURBATIONS}
            for cr, pdict in results:
                complex_rows.extend(cr)
                for pname in PERTURBATIONS:
                    perturbed_examples[pname].extend(pdict[pname])
        else:
            print("Setting up Stanza parser...")
            nlp = setup_parser()
            complex_rows = []
            perturbed_examples = {pname: [] for pname in PERTURBATIONS}
            for ex in tqdm(examples, desc="Perturbing"):
                prem, hyp = ex['premise'], ex['hypothesis']
                prem_doc = nlp(prem)
                hyp_doc = nlp(hyp)
                orig_text = f"{prem} {hyp}"
                orig_complex = compute_complexity(orig_text, nlp)
                for metric, val in orig_complex.items():
                    complex_rows.append({'perturbation method': 'original', 'metric type': metric, 'value': val})
                for pname, pfunc in PERTURBATIONS.items():
                    pert_prem = pfunc(prem, nlp, doc=prem_doc)
                    pert_hyp = pfunc(hyp, nlp, doc=hyp_doc)
                    pert_text = f"{pert_prem} {pert_hyp}"
                    pert_complex = compute_complexity(pert_text, nlp)
                    perturbed_examples[pname].append({'premise': pert_prem, 'hypothesis': pert_hyp, 'text': pert_text, 'label': ex['label'], 'split': ex['split']})
                    for metric, val in pert_complex.items():
                        complex_rows.append({'perturbation method': pname, 'metric type': metric, 'value': val})

        complex_df = pd.DataFrame(complex_rows)
        complex_agg = complex_df.groupby(['perturbation method', 'metric type']).agg({'value': 'mean'}).reset_index()
        complex_agg.to_csv(str(out_dir / 'complex.csv'), index=False)
        print("Saved complex.csv")

        # Save cache for future --data_path runs
        with open(cache_file, 'w') as f:
            json.dump({'perturbed': perturbed_examples, 'complex_rows': complex_rows}, f)
        print(f"Saved perturbation cache to {cache_file}")

    # Part 2 & 4: Load models and evaluate
    models = [
        'textattack/bert-base-uncased-MNLI',
        'textattack/roberta-base-MNLI',
        'microsoft/deberta-base-mnli',
        'typeform/distilbert-base-uncased-mnli',
    ]

    perf_rows = []
    for model_name in models:
        print(f"Loading {model_name}...")
        pipe = get_nli_pipeline(model_name)

        # Baseline (original, unperturbed)
        acc = evaluate_model(pipe, examples, perturbation_name='original')
        perf_rows.append({'model': model_name, 'perturbation method': 'original', 'performance': acc})
        print(f"  Original: {acc:.4f}")

        for pname in PERTURBATIONS:
            acc_p = evaluate_model(pipe, examples, perturbed_list=perturbed_examples[pname], perturbation_name=pname)
            perf_rows.append({'model': model_name, 'perturbation method': pname, 'performance': acc_p})
            print(f"  {pname}: {acc_p:.4f}")

    perf_df = pd.DataFrame(perf_rows)
    perf_df.to_csv(str(out_dir / 'perf.csv'), index=False)
    print("Saved perf.csv")
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_size', type=int, default=15000, help='Number of examples to sample from MultiNLI')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory for data.jsonl, perf.csv, complex.csv')
    parser.add_argument('--data_path', type=str, default=None, help='Use existing data.jsonl instead of loading from HuggingFace')
    parser.add_argument('--workers', type=int, default=1, help='Parallel workers for perturbation (e.g., 4 for ~4x speedup)')
    args = parser.parse_args()
    main(args)
