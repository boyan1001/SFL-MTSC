import json
import argparse
import sys
import re
from collections import defaultdict


def normalize_text(text):
    """
    Normalizes text:
    1. Convert to lowercase.
    2. Convert Chinese numerals to Arabic numerals.
    3. Remove punctuation.
    4. Remove extra whitespace.
    """
    if not isinstance(text, str):
        return str(text)

    text = text.lower()

    cn_num_map = {
        '零': '0', '一': '1', '二': '2', '三': '3', '四': '4',
        '五': '5', '六': '6', '七': '7', '八': '8', '九': '9',
        '两': '2',
    }
    for k, v in cn_num_map.items():
        text = text.replace(k, v)

    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


def normalize_semantics(semantics_list):
    """
    Normalizes all string fields in a semantics list.
    Returns a list of dicts with normalized domain, intent, and slots.
    """
    if not isinstance(semantics_list, list):
        return []

    normalized = []
    for item in semantics_list:
        if not isinstance(item, dict):
            continue

        new_item = {}
        if 'domain' in item:
            new_item['domain'] = normalize_text(item['domain'])
        if 'intent' in item:
            new_item['intent'] = normalize_text(item['intent'])

        if 'slots' in item:
            slots = item['slots']
            if isinstance(slots, dict):
                new_item['slots'] = {
                    normalize_text(k): normalize_text(v)
                    for k, v in slots.items()
                }
            else:
                new_item['slots'] = slots

        normalized.append(new_item)

    return normalized


def extract_slot_set(semantics):
    """
    Flattens a normalized semantics list into a set of (key, value) slot pairs.
    """
    slot_set = set()
    for frame in semantics:
        slots = frame.get('slots', {})
        if isinstance(slots, dict):
            for k, v in slots.items():
                slot_set.add((k, v))
    return slot_set


def classify_wrong_slots(pred_slot_set, gt_slot_set):
    """
    Among false positive slot pairs, classifies each as:
    - wrong_value: slot key exists in GT but with a different value
    - wrong_name: slot value exists in GT but under a different key
    Returns (has_wrong_name, has_wrong_value).
    """
    gt_values_by_name = defaultdict(set)
    gt_names_by_value = defaultdict(set)
    for k, v in gt_slot_set:
        gt_values_by_name[k].add(v)
        gt_names_by_value[v].add(k)

    has_wrong_name = False
    has_wrong_value = False

    for k_pred, v_pred in pred_slot_set - gt_slot_set:
        if k_pred in gt_values_by_name and v_pred not in gt_values_by_name[k_pred]:
            has_wrong_value = True
        elif v_pred in gt_names_by_value and k_pred not in gt_names_by_value[v_pred]:
            has_wrong_name = True

    return has_wrong_name, has_wrong_value


def load_ground_truth(gt_file):
    gt_map = {}
    with open(gt_file, 'r', encoding='utf-8', errors='replace') as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                sample_id = str(data.get('id', ''))
                if not sample_id:
                    print(
                        f"Warning: Ground truth line {line_idx} missing 'id', skipped.",
                        file=sys.stderr,
                    )
                    continue
                gt_map[sample_id] = data
            except json.JSONDecodeError:
                print(
                    f"Warning: JSON parse error at ground truth line {line_idx}, skipped.",
                    file=sys.stderr,
                )
    return gt_map


def calculate_metrics(predict_file, ground_truth_file):
    try:
        gt_map = load_ground_truth(ground_truth_file)
    except FileNotFoundError as e:
        print(f"Error: Ground truth file not found - {e}", file=sys.stderr)
        sys.exit(1)

    overall_match = 0
    intent_match = 0
    slot_tp = slot_fp = slot_fn = 0
    processed = 0
    wrong_slot = 0
    wrong_slot_name = 0
    wrong_slot_value = 0

    try:
        with open(predict_file, 'r', encoding='utf-8', errors='replace') as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    pred_data = json.loads(line)
                except json.JSONDecodeError:
                    print(
                        f"Warning: JSON parse error at prediction line {line_idx}, skipped.",
                        file=sys.stderr,
                    )
                    continue

                sample_id = str(pred_data.get('id', ''))
                if sample_id not in gt_map:
                    continue

                gt_data = gt_map[sample_id]
                processed += 1

                pred_sem = normalize_semantics(pred_data.get('semantics', []))
                gt_sem = normalize_semantics(gt_data.get('semantics', []))

                # Overall Accuracy: exact match on full semantics
                if pred_sem == gt_sem:
                    overall_match += 1

                # Intent Accuracy: all (domain, intent) pairs match
                pred_intents = sorted(
                    (s.get('domain'), s.get('intent')) for s in pred_sem
                )
                gt_intents = sorted(
                    (s.get('domain'), s.get('intent')) for s in gt_sem
                )
                if pred_intents == gt_intents:
                    intent_match += 1

                # Slot F1: skip samples with wrong slot name/value; track stats
                pred_slots = extract_slot_set(pred_sem)
                gt_slots = extract_slot_set(gt_sem)

                has_wrong_name, has_wrong_value = classify_wrong_slots(pred_slots, gt_slots)

                if has_wrong_name or has_wrong_value:
                    wrong_slot += 1
                    if has_wrong_name:
                        wrong_slot_name += 1
                    if has_wrong_value:
                        wrong_slot_value += 1
                    continue

                slot_tp += len(pred_slots & gt_slots)
                slot_fp += len(pred_slots - gt_slots)
                slot_fn += len(gt_slots - pred_slots)

    except FileNotFoundError as e:
        print(f"Error: Prediction file not found - {e}", file=sys.stderr)
        sys.exit(1)

    if processed == 0:
        print("Error: No matching Sample IDs found between files.", file=sys.stderr)
        return {k: 0.0 for k in ['overall_accuracy', 'intent_accuracy', 'slot_f1']}

    overall_accuracy = overall_match / processed
    intent_accuracy = intent_match / processed

    slot_precision = slot_tp / (slot_tp + slot_fp) if (slot_tp + slot_fp) > 0 else 0.0
    slot_recall = slot_tp / (slot_tp + slot_fn) if (slot_tp + slot_fn) > 0 else 0.0
    slot_f1 = (
        2 * slot_precision * slot_recall / (slot_precision + slot_recall)
        if (slot_precision + slot_recall) > 0
        else 0.0
    )

    return {
        'total_count': processed,
        'overall_match_count': overall_match,
        'overall_accuracy': overall_accuracy,
        'intent_match_count': intent_match,
        'intent_accuracy': intent_accuracy,
        'slot_tp': slot_tp,
        'slot_fp': slot_fp,
        'slot_fn': slot_fn,
        'slot_precision': slot_precision,
        'slot_recall': slot_recall,
        'slot_f1': slot_f1,
        'wrong_slot': wrong_slot,
        'wrong_slot_rate': wrong_slot / processed,
        'wrong_slot_name': wrong_slot_name,
        'wrong_slot_name_rate': wrong_slot_name / processed,
        'wrong_slot_value': wrong_slot_value,
        'wrong_slot_value_rate': wrong_slot_value / processed,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Calculate NLU Evaluation Metrics (multi-intent, normalized)',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('predict_file', help='Path to prediction .jsonl file')
    parser.add_argument('ground_truth_file', help='Path to ground truth .jsonl file')
    args = parser.parse_args()

    results = calculate_metrics(args.predict_file, args.ground_truth_file)

    print('-' * 60)
    print('Evaluation Results (Normalization: case / punct / numerals)')
    print('-' * 60)
    print(f"Total Records Processed : {results['total_count']}")

    print('\n--- Overall Accuracy (Exact Semantics Match) ---')
    print(f"Exact Matches           : {results['overall_match_count']}")
    print(f"Accuracy                : {results['overall_accuracy']:.4f}  ({results['overall_accuracy']:.2%})")

    print('\n--- Intent Accuracy (All Intents Correct) ---')
    print(f"Intent Matches          : {results['intent_match_count']}")
    print(f"Accuracy                : {results['intent_accuracy']:.4f}  ({results['intent_accuracy']:.2%})")

    print('\n--- Slot Filling F1 (Global Aggregation) ---')
    print(f"TP / FP / FN            : {results['slot_tp']} / {results['slot_fp']} / {results['slot_fn']}")
    print(f"Precision               : {results['slot_precision']:.4f}")
    print(f"Recall                  : {results['slot_recall']:.4f}")
    print(f"F1 Score                : {results['slot_f1']:.4f}  ({results['slot_f1']:.2%})")

    print('\n--- Wrong Slot Statistics (excluded from F1) ---')
    print(f"Wrong Slot Samples      : {results['wrong_slot']}  ({results['wrong_slot_rate']:.2%})")
    print(f"  - Wrong Slot Name     : {results['wrong_slot_name']}  ({results['wrong_slot_name_rate']:.2%})")
    print(f"  - Wrong Slot Value    : {results['wrong_slot_value']}  ({results['wrong_slot_value_rate']:.2%})")
    print('-' * 60)


if __name__ == '__main__':
    main()