import json
import argparse
import sys
import re
import os
import logging
import math

from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, deque
from typing import Any, Dict, List, Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def setup_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process Multi-intent Multi-task Self-consistency")
    
    # input dir
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing the deffetent reasoning paths result JSONL files")

    # output file
    parser.add_argument("--output-file", type=str, required=True, help="Path to the SFL-MTSC result output JSONL file")

    # hyperparameters
    parser.add_argument("--alpha", type=float, default=0.3, help="Hybrid Jaccard alpha (weight for key-value Jaccard)")
    parser.add_argument("--tau", type=float, default=0.55, help="Similarity threshold for clustering")

    return parser

def normalize_slot_key(text: str) -> str:
    if '-' in text:
        return text.split('-', 1)[1].strip()
    return text.strip()

# --------------------
# Frame Clustering
# --------------------

## slot processing
def slot_pair(frame) -> Set[Tuple[str, str]]:
    slots = frame.get("slots", {}) or {}
    if not isinstance(slots, dict):
        return set()
    return {(str(k), str(v)) for k, v in slots.items()}

def slot_values(frame) -> Set[str]:
    slots = frame.get("slots", {}) or {}
    if not isinstance(slots, dict):
        return set()
    return {str(v) for v in slots.values()}

## Jaccard
def jaccard(a: Set[Any], b: Set[Any]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def hybrid_jaccard(fa, fb, alpha=0.3) -> float:
    sa = slot_pair(fa)
    sb = slot_pair(fb)
    va = slot_values(fa)
    vb = slot_values(fb)

    j_kv = jaccard(sa, sb)
    j_val = jaccard(va, vb)

    sim_hyb = alpha * j_kv + (1 - alpha) * j_val
    return sim_hyb


def cluster_hybrid_jaccard(frames, alpha=0.3, tau=0.55):
    # 0) ignore some case
    n = len(frames)
    if n == 0:
        return []
    if n == 1:
        return [frames[:]]

    # 1) Build similarity graph
    adj: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            sim = hybrid_jaccard(frames[i], frames[j], alpha=alpha)
            if sim >= tau:
                adj[i].append(j)
                adj[j].append(i)

    # 2) Connected components
    visited = [False] * n
    cluster: List[List[Dict[str, Any]]] = []
    for i in range(n):
        if visited[i]:
            continue
        queue = deque([i])
        visited[i] = True
        comp = []

        while queue:
            u = queue.popleft()
            comp.append(frames[u])
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)

        cluster.append(comp)

    return cluster

# --------------------
# Self-Consistency Score
# --------------------
def compute_support(cluster: List[Dict]) -> int:
    path_indices = set()
    for frame in cluster:
        p = frame.get("_path_idx")
        if p is not None:
            path_indices.add(p)
    return len(path_indices)

# --------------------
# Filtering
# --------------------
def filter_di_clusters(di_cluster, k):
    sup_threshold = math.ceil(k / 2)

    retained = defaultdict(list)
    for di_key, backet in di_cluster.items():
        support = compute_support(backet)
        if support >= sup_threshold:
            retained[di_key] = backet
    return retained

def filter_slot_clusters(slot_clusters, k):
    sup_threshold = math.ceil(k / 2)

    retained = defaultdict(list)
    for di_key, clusters in slot_clusters.items():
        for cluster in clusters:
            support = compute_support(cluster)
            if support >= sup_threshold:
                retained[di_key].append(cluster)
    return retained

# --------------------
# Re-integration
# --------------------
def voting_value(cluster: List[Dict]):
    final_slots = {}

    all_keys: Set[str] = set()
    for frame in cluster:
        slots = frame.get("slots", {}) or {}
        if isinstance(slots, dict):
            all_keys.update(slots.keys())

    for key in all_keys:
        value_counts = defaultdict(int)
        for frame in cluster:
            slots = frame.get("slots", {}) or {}
            if isinstance(slots, dict) and key in slots:
                value_counts[str(slots[key])] += 1

        if not value_counts:
            continue
    
        best_value = max(value_counts, key=lambda v: (value_counts[v], [-ord(c) for c in v]))
        final_slots[key] = best_value
    return final_slots

def voting_key(cluster: List[Dict]):
    value_key_counts = defaultdict(lambda: defaultdict(int))
    value_support = defaultdict(set)

    # 1) Collect (key, value) pairs and count support for each value
    for frame_idx, frame in enumerate(cluster):
        slots = frame.get("slots", {}) or {}
        if not isinstance(slots, dict):
            continue
        for key, value in slots.items():
            value = str(value)
            value_key_counts[value][key] += 1
            value_support[value].add(frame_idx)

    # 2) Filter values by support
    sup_threshold = math.ceil(len(cluster) / 2)
    retained_values = {v for v, sup in value_support.items() if len(sup) >= sup_threshold}

    # 3) Voting key for each retained value
    final_slots = {}
    for value in retained_values:
        key_counts = value_key_counts[value]
        best_key = max(key_counts, key=lambda k: (key_counts[k], [-ord(c) for c in k]))
        final_slots[best_key] = value

    return final_slots


def reintegrate_cluster(di_key: Tuple, cluster: List[Dict]) -> Dict:
    domain, intent = di_key
    final_slots = voting_key(cluster)

    return {
        "domain": domain,
        "intent": intent,
        "slots": final_slots
    }
    

def SFL_MTSC(semantics, alpha=0.3, tau=0.55):
    k = len(semantics)
    if k == 0:
        return []
    if k == 1:
        return semantics[0]

    # 1) Collect Frame Pool
    frame_pool = []
    for path_idx, semantics_list in enumerate(semantics):
        for frame in semantics_list:
            if not isinstance(frame, dict):
                logging.warning(f"Skipping non-dict frame at path {path_idx}: {type(frame)} = {frame}")
                continue
            if not frame.get("domain") or not frame.get("intent"):
                continue
            frame_with_path = dict(frame)
            frame_with_path["_path_idx"] = path_idx
            frame_pool.append(frame_with_path)

    # 2) Clustering
    ## 2-1) Cluster by (domain, intent)
    di_cluster = defaultdict(list)
    for frame in frame_pool:
        domain = frame.get("domain", "")
        intent = frame.get("intent", "")
        di_cluster[(domain, intent)].append(frame)

    ## 2-2) Cluster by slot Jaccard similarity
    slot_clusters = {}
    for di_key, frames in di_cluster.items():
        slot_clusters[di_key] = cluster_hybrid_jaccard(
            frames,
            alpha=alpha,
            tau=tau
        )

    # 4 + 5) Score & Filter
    retained_slot_cluster = filter_slot_clusters(slot_clusters, k=k)

    # 6) Re-integration
    final_semantics = []
    for di_key, clusters in retained_slot_cluster.items():
        for cluster in clusters:
            representative_frame = reintegrate_cluster(di_key, cluster)
            representative_frame.pop("_path_idx", None)
            final_semantics.append(representative_frame)

    return final_semantics

def process_files(args: argparse.Namespace):
    # 1) Prefare file paths
    input_dir = Path(args.input_dir)
    all_input_files = list(input_dir.glob("*.jsonl"))
    if input_dir / "SFL-MTSC.jsonl" in all_input_files:
        all_input_files.remove(input_dir / "SFL-MTSC.jsonl")
    logging.info(f"Found {len(all_input_files)} JSONL files in the input directory.")
    
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 2) Integrate the reasoning paths results for each sample and perform self-consistency voting
    sample_results = defaultdict(list)
    id_to_query = defaultdict(str)
    for input_file in tqdm(all_input_files, desc="Processing input files"):
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                sample_id = data.get("id", "")
                query = data.get("query", "")
                semantics = data.get("semantics", "")

                # normalize semantics slot keys
                if isinstance(semantics, list):
                    for frame in semantics:
                        if "slots" in frame and isinstance(frame["slots"], dict):
                            normalized_slots = {}
                            for k, v in frame["slots"].items():
                                norm_k = normalize_slot_key(k)
                                normalized_slots[norm_k] = v
                            frame["slots"] = normalized_slots
                
                if sample_id != "":
                    sample_results[sample_id].append(semantics)
                    id_to_query[sample_id] = query

    # 3) Perform SFL-MTSC and get final predictions for each sample
    logging.info(f"Starting SFL-MTSC processing for {len(sample_results)} samples.")
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for sample_id, semantics_list in tqdm(sample_results.items(), desc="Performing SFL-MTSC"):
            final_semantics = SFL_MTSC(
                semantics_list,
                alpha=args.alpha,
                tau=args.tau,
            )
            final_pred = {
                "id": sample_id,
                "query": id_to_query.get(sample_id, ""),
                "semantics": final_semantics
            }
    
            f_out.write(json.dumps(final_pred, ensure_ascii=False) + "\n")

    logging.info(f"SFL-MTSC processing completed. Final results saved to {output_file}.")

def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    if not args.input_dir.endswith('/'):
        args.input_dir += '/'

    process_files(args)

if __name__ == "__main__":
    main()