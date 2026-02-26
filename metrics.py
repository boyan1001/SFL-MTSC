import json
import argparse
import sys
import re
from collections import defaultdict

def normalize_text(text):
    """
    Normalizes text:
    1. Convert to lowercase.
    2. Convert Chinese numerals to Arabic numerals (character-level replacement).
    3. Remove punctuation.
    4. Remove extra whitespace.
    """
    if not isinstance(text, str):
        return str(text)

    # 1. Convert to lowercase
    text = text.lower()

    # 2. Chinese numeral mapping (Simple character replacement)
    # KEEPING CHINESE CHARACTERS HERE AS REQUESTED
    cn_num_map = {
        '零': '0', '一': '1', '二': '2', '三': '3', '四': '4',
        '五': '5', '六': '6', '七': '7', '八': '8', '九': '9',
        '两': '2'
    }
    for k, v in cn_num_map.items():
        text = text.replace(k, v)

    # 3. Remove punctuation
    # Logic: Replace any character that is NOT a word char, digit, or whitespace with empty string.
    # \w in Python 3 re includes alphanumeric characters (including Chinese characters) and underscores.
    text = re.sub(r'[^\w\s]', '', text)

    # 4. Remove leading/trailing whitespace
    return text.strip()

def normalize_semantics(semantics_list):
    """
    Recursively normalizes all key fields in the semantics list.
    """
    if not isinstance(semantics_list, list):
        return []

    normalized_list = []
    for item in semantics_list:
        if not isinstance(item, dict):
            continue
            
        new_item = {}
        
        # Normalize domain and intent
        if 'domain' in item:
            new_item['domain'] = normalize_text(item['domain'])
        if 'intent' in item:
            new_item['intent'] = normalize_text(item['intent'])
            
        # Normalize slots
        if 'slots' in item:
            origin_slots = item['slots']
            if isinstance(origin_slots, dict):
                new_slots = {}
                for k, v in origin_slots.items():
                    # Normalize both Slot Key and Value
                    norm_k = normalize_text(k)
                    norm_v = normalize_text(v)
                    new_slots[norm_k] = norm_v
                new_item['slots'] = new_slots
            else:
                new_item['slots'] = origin_slots
        
        normalized_list.append(new_item)
    
    return normalized_list

def calculate_metrics(predict_file, ground_truth_file):
    """
    修改版本：支援 ID 匹配，自動跳過缺失或不對齊的數據。
    """
    # 1. 讀取 Ground Truth 並建立索引
    gt_map = {}
    try:
        # 使用 errors='replace' 防止非法位元組導致崩潰
        with open(ground_truth_file, 'r', encoding='utf-8', errors='replace') as f_gt:
            for line_idx, line in enumerate(f_gt, 1):
                try:
                    data = json.loads(line.strip())
                    sample_id = str(data.get("id", ""))
                    if not sample_id:
                        print(f"Warning: Ground truth line {line_idx} missing 'id', skipped.", file=sys.stderr)
                        continue
                    gt_map[sample_id] = data
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError as e:
        print(f"Error: Ground truth file not found - {e}", file=sys.stderr)
        sys.exit(1)

    # 2. 迭代處理 Prediction 檔案
    overall_match_count = 0
    intent_match_count = 0
    slot_tp, slot_fp, slot_fn = 0, 0, 0 
    processed_count = 0
    wrong_slot_name = 0
    wrong_slot_value = 0
    wrong_slot = 0

    try:
        with open(predict_file, 'r', encoding='utf-8', errors='replace') as f_pred:
            for line_idx, pred_line in enumerate(f_pred, 1):
                try:
                    pred_data = json.loads(pred_line.strip())
                    sample_id = str(pred_data.get("id", ""))
                    
                    # 檢查 ID 是否存在於 Ground Truth 中
                    if sample_id not in gt_map:
                        # 這裡直接跳過，不中斷程式
                        continue
                    
                    gt_data = gt_map[sample_id]
                    processed_count += 1

                    # --- 核心邏輯保持不變 ---
                    pred_semantics = normalize_semantics(pred_data.get("semantics", []))
                    gt_semantics = normalize_semantics(gt_data.get("semantics", []))
                    
                    # 1. Overall Accuracy
                    if pred_semantics == gt_semantics:
                        overall_match_count += 1
                    
                    # 2. Intent Accuracy
                    pred_intents = sorted([(s.get("domain"), s.get("intent")) for s in pred_semantics])
                    gt_intents = sorted([(s.get("domain"), s.get("intent")) for s in gt_semantics])
                    if pred_intents == gt_intents:
                        intent_match_count += 1

                    # 3. Slot Metrics
                    # - 车载控制-action: 车载控制中的具体动作，例如调节。
                    # - 车载控制-操作: 对车载功能进行的操作，例如关闭、设置、转到。
                    # - 车载控制-object: 车载控制的对象，例如座椅。
                    # - 车载控制-对象: 车载控制的具体对象，例如空调、车内灯、阅读灯。

                    pred_slot_set = set()
                    for s in pred_semantics:
                        slots = s.get("slots", {})

                        # convert english to chinese for slot keys
                        new_slots = {
                            "action": "操作",
                            "object": "对象",
                        }

                        if isinstance(slots, dict):
                            for k, v in slots.items():
                                if k in new_slots:
                                    k = new_slots[k]
                                pred_slot_set.add((k, v))
                    
                    gt_slot_set = set()
                    wrong_slot_name_count = 0
                    wrong_slot_value_count = 0
                    total_pred_slots = 0

                    for s in gt_semantics:
                        slots = s.get("slots", {})
                        if isinstance(slots, dict):
                            for k, v in slots.items():
                                gt_slot_set.add((k, v))

                    gt_values_by_name = defaultdict(set)
                    gt_names_by_value = defaultdict(set)

                    for k, v in gt_slot_set:
                        gt_values_by_name[k].add(v)
                        gt_names_by_value[v].add(k)

                    fp_pairs = pred_slot_set.difference(gt_slot_set)

                    for (k_pred, v_pred) in fp_pairs:
                        # wrong slot value: name exists in GT but value mismatched
                        if k_pred in gt_values_by_name and v_pred not in gt_values_by_name[k_pred]:
                            wrong_slot_value_count += 1
                            continue

                        # wrong slot name: value exists in GT but name mismatched
                        if v_pred in gt_names_by_value and k_pred not in gt_names_by_value[v_pred]:
                            wrong_slot_name_count += 1
                            continue

                    if wrong_slot_name_count > 0 or wrong_slot_value_count > 0:
                        wrong_slot += 1
                        if wrong_slot_name_count > 0:
                            wrong_slot_name += 1
                        if wrong_slot_value_count > 0:
                            wrong_slot_value += 1
                        continue

                    slot_tp += len(pred_slot_set.intersection(gt_slot_set))
                    slot_fp += len(pred_slot_set.difference(gt_slot_set))
                    slot_fn += len(gt_slot_set.difference(pred_slot_set))

                except json.JSONDecodeError:
                    print(f"Warning: JSON parse error at prediction line {line_idx}, skipped.", file=sys.stderr)
                except Exception as e:
                    print(f"Warning: Unexpected error at line {line_idx}: {e}", file=sys.stderr)

    except FileNotFoundError as e:
        print(f"Error: Prediction file not found - {e}", file=sys.stderr)
        sys.exit(1)

    # 檢查是否有成功匹配的數據
    if processed_count == 0:
        print("Error: No matching Sample IDs found between files.", file=sys.stderr)
        return {k: 0.0 for k in ["overall_accuracy", "intent_accuracy", "slot_f1"]} # 簡化返回

    # --- 計算最終指標 (除數改為 processed_count) ---
    overall_accuracy = overall_match_count / processed_count
    intent_accuracy = intent_match_count / processed_count

    slot_precision = slot_tp / (slot_tp + slot_fp) if (slot_tp + slot_fp) > 0 else 0.0
    slot_recall = slot_tp / (slot_tp + slot_fn) if (slot_tp + slot_fn) > 0 else 0.0
    slot_f1 = 2 * (slot_precision * slot_recall) / (slot_precision + slot_recall) if (slot_precision + slot_recall) > 0 else 0.0

    wrong_slot_rate = wrong_slot / processed_count if processed_count > 0 else 0.0
    wrong_slot_name_rate = wrong_slot_name / processed_count if processed_count > 0 else 0.0
    wrong_slot_value_rate = wrong_slot_value / processed_count if processed_count > 0 else 0.0

    return {
        "total_count": processed_count, # 這是實際有對齊到的樣本數
        "overall_match_count": overall_match_count,
        "overall_accuracy": overall_accuracy,
        "intent_match_count": intent_match_count,
        "intent_accuracy": intent_accuracy,
        "slot_tp": slot_tp, "slot_fp": slot_fp, "slot_fn": slot_fn,
        "slot_precision": slot_precision,
        "slot_recall": slot_recall,
        "slot_f1": slot_f1,
        "wrong_slot_rate": wrong_slot_rate,
        "wrong_slot_name_rate": wrong_slot_name_rate,
        "wrong_slot_value_rate": wrong_slot_value_rate,
    }

def main():
    parser = argparse.ArgumentParser(
        description="Calculate NLU Evaluation Metrics (Multi-intent & Normalization supported)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("predict_file", help="Path to prediction .jsonl file")
    parser.add_argument("ground_truth_file", help="Path to ground truth .jsonl file")
    args = parser.parse_args()

    results = calculate_metrics(args.predict_file, args.ground_truth_file)

    print("-" * 60)
    print(f"Evaluation Results (Normalization Enabled: Case/Punct/Num)")
    print("-" * 60)
    print(f"Total Records Processed: {results['total_count']}")
    
    print("\n--- Overall Accuracy (Semantics Exact Match) ---")
    print(f"Exact Matches:    {results['overall_match_count']}")
    print(f"Accuracy:         {results['overall_accuracy']:.4f} ({results['overall_accuracy']:.2%})")
    
    print("\n--- Intent Accuracy (All Intents Correct) ---")
    print(f"Intent Matches:   {results['intent_match_count']}")
    print(f"Accuracy:         {results['intent_accuracy']:.4f} ({results['intent_accuracy']:.2%})")

    print("\n--- Slot Filling F1-Score (Global Aggregation) ---")
    print(f"TP / FP / FN:     {results['slot_tp']} / {results['slot_fp']} / {results['slot_fn']}")
    print(f"Precision:        {results['slot_precision']:.4f}")
    print(f"Recall:           {results['slot_recall']:.4f}")
    print(f"F1 Score:         {results['slot_f1']:.4f} ({results['slot_f1']:.2%})")
    print(f"Wrong Slot Rate:  {results['wrong_slot_rate']:.4f} ({results['wrong_slot_rate']:.2%})")
    print(f"Wrong Slot Name:  {results['wrong_slot_name_rate']:.4f} ({results['wrong_slot_name_rate']:.2%})")
    print(f"Wrong Slot Value: {results['wrong_slot_value_rate']:.4f} ({results['wrong_slot_value_rate']:.2%})")
    print("-" * 60)

if __name__ == "__main__":
    main()