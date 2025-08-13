#!/usr/bin/env python3
"""
Analyze tokenized timelines in a processed_data directory and create a histogram
of timeline lengths, along with basic summary statistics.

Usage:
  python analyze_timelines.py --data_dir processed_data_<tag> \
      --bins 50 --logy --save_dir plots

Outputs:
  - Histogram PNG saved to <save_dir>/timeline_length_hist_<tag>.png
  - Summary text saved to <save_dir>/timeline_stats_<tag>.txt
  - Optional CSV with per-patient lengths
"""

import os
import argparse
import pickle
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt


def load_timelines(data_dir: str) -> Dict[int, List[int]]:
    path = os.path.join(data_dir, 'tokenized_timelines.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(f"tokenized_timelines.pkl not found in {data_dir}")
    with open(path, 'rb') as f:
        timelines = pickle.load(f)
    return timelines


def summarize_lengths(lengths: List[int]) -> str:
    arr = np.asarray(lengths, dtype=np.int64)
    lines = []
    lines.append(f"Patients: {arr.size}")
    lines.append(f"Min length: {arr.min() if arr.size else 0}")
    lines.append(f"Max length: {arr.max() if arr.size else 0}")
    if arr.size:
        lines.append(f"Mean length: {arr.mean():.2f}")
        lines.append(f"Median length: {np.median(arr):.2f}")
        lines.append(
            "Percentiles (p10/p25/p75/p90): "
            f"{np.percentile(arr, 10):.0f} / {np.percentile(arr, 25):.0f} / "
            f"{np.percentile(arr, 75):.0f} / {np.percentile(arr, 90):.0f}"
        )
    # Bucketed counts like prior prints
    buckets = {'0-10':0,'10-20':0,'20-100':0,'100-200':0,'200-800':0,'>800':0}
    for n in arr:
        if n <= 10:
            buckets['0-10'] += 1
        elif n <= 20:
            buckets['10-20'] += 1
        elif n <= 100:
            buckets['20-100'] += 1
        elif n <= 200:
            buckets['100-200'] += 1
        elif n <= 800:
            buckets['200-800'] += 1
        else:
            buckets['>800'] += 1
    lines.append("Bucketed counts:")
    for k in ['0-10','10-20','20-100','100-200','200-800','>800']:
        lines.append(f"  {k}: {buckets[k]}")
    return "\n".join(lines)


def make_histogram(lengths: List[int], bins: int, title: str, logy: bool, out_path: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=bins, color='#4e79a7', edgecolor='white')
    plt.title(title)
    plt.xlabel('Timeline length (tokens)')
    plt.ylabel('Count')
    if logy:
        plt.yscale('log')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze processed timelines and plot length histogram')
    parser.add_argument('--data_dir', required=True, help='Processed data directory (contains tokenized_timelines.pkl)')
    parser.add_argument('--save_dir', default='plots', help='Directory to save outputs (default: plots)')
    parser.add_argument('--bins', type=int, default=50, help='Number of histogram bins (default: 50)')
    parser.add_argument('--logy', action='store_true', help='Log scale y-axis for histogram')
    parser.add_argument('--save_csv', action='store_true', help='Also save per-patient lengths as CSV')
    args = parser.parse_args()

    timelines = load_timelines(args.data_dir)
    lengths = [len(seq) for seq in timelines.values()]

    tag = os.path.basename(os.path.abspath(args.data_dir))
    if tag.startswith('processed_data_'):
        tag = tag[len('processed_data_'):]

    # Summary
    summary = summarize_lengths(lengths)
    print('\nTimeline length summary:')
    print(summary)

    # Save summary
    os.makedirs(args.save_dir, exist_ok=True)
    stats_path = os.path.join(args.save_dir, f'timeline_stats_{tag}.txt')
    with open(stats_path, 'w') as f:
        f.write(summary + '\n')

    # Histogram
    hist_path = os.path.join(args.save_dir, f'timeline_length_hist_{tag}.png')
    make_histogram(lengths, bins=args.bins, title=f'Timeline lengths ({tag})', logy=args.logy, out_path=hist_path)
    print(f"Saved histogram to: {hist_path}")
    print(f"Saved summary to:   {stats_path}")

    # Optional CSV
    if args.save_csv:
        import csv
        csv_path = os.path.join(args.save_dir, f'timeline_lengths_{tag}.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['patient_id', 'length'])
            for pid, seq in timelines.items():
                w.writerow([pid, len(seq)])
        print(f"Saved lengths CSV to: {csv_path}")


if __name__ == '__main__':
    main()


