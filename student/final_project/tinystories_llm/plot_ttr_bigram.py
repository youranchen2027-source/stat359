#!/usr/bin/env python3
"""
仅根据 overall_summary.json 绘制三种风格的：
  - avg_type_token_ratio
  - corpus_type_token_ratio
  - avg_repeated_bigram_count
保存到 evaluation_results_v1/figures/voice_characteristics/。
依赖: pip install matplotlib numpy
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "evaluation_results_v1"
OUT_DIR = RESULTS_DIR / "figures" / "voice_characteristics"


def main():
    summary_path = RESULTS_DIR / "overall_summary.json"
    if not summary_path.exists():
        print(f"未找到: {summary_path}")
        return
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    personas = list(summary.keys())

    configs = [
        ("avg_type_token_ratio", "Avg Type-Token Ratio by Persona", "Avg Type-Token Ratio", "avg_type_token_ratio.png"),
        ("corpus_type_token_ratio", "Corpus Type-Token Ratio by Persona", "Corpus Type-Token Ratio", "corpus_type_token_ratio.png"),
        ("avg_repeated_bigram_count", "Avg Repeated Bigram Count by Persona", "Avg Repeated Bigram Count", "avg_repeated_bigram_count.png"),
    ]

    for metric_key, title, ylabel, fname in configs:
        vals = []
        for p in personas:
            vc = summary[p].get("voice_characteristics") or {}
            vals.append(vc.get(metric_key, 0))

        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(personas))
        colors = ["#3498db", "#2ecc71", "#e74c3c"]
        bars = ax.bar(x, vals, color=colors[: len(personas)], edgecolor="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([p.capitalize() for p in personas])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        max_val = max(vals) if vals else 0
        for bar, v in zip(bars, vals):
            offset = 0.01 * max_val if max_val > 0 else 0.02
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                    f"{v:.4f}" if abs(v) < 1 else f"{v:.2f}",
                    ha="center", va="bottom", fontsize=10)
        plt.tight_layout()
        out_path = OUT_DIR / fname
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")

    print(f"三张图已保存到: {OUT_DIR}")


if __name__ == "__main__":
    main()
