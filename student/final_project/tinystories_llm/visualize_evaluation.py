#!/usr/bin/env python3
"""
可视化 evaluation_results 与 tinystories_model 相关结果。

- evaluation_results: 各 persona 的评估汇总、persona 一致性、任务质量、失败/错误分布等
- tinystories_model: 若存在 TensorBoard 日志，绘制训练 loss/perplexity 曲线；若存在 checkpoint-*.pth，可绘制稀疏的「按 checkpoint 的 loss」图；若有 args.json 则输出配置概览图。model_epoch_*.pth 仅含权重，无法直接绘 loss 曲线。

依赖: pip install matplotlib numpy  （可选: pip install tensorboard 读取训练曲线；可选: torch 读取 checkpoint 绘制稀疏 loss）

用法:
  python visualize_evaluation.py
  python visualize_evaluation.py --results_dir ./evaluation_results --model_dir ./tinystories_model --out_dir ./figures
"""

import json
import os
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 可选：读取 TensorBoard 事件以绘制训练曲线
try:
    from tensorboard.backend.event_processing import event_accumulator
    HAS_TB = True
except ImportError:
    HAS_TB = False

# 可选：从 checkpoint-*.pth 读取 step/loss 绘制稀疏曲线
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------- 路径与配置 ----------
def parse_args():
    p = argparse.ArgumentParser(description="Visualize evaluation_results and tinystories_model")
    p.add_argument("--results_dir", type=str, default=None,
                   help="evaluation_results 目录，默认与脚本同级的 evaluation_results")
    p.add_argument("--model_dir", type=str, default=None,
                   help="tinystories_model 目录（含 TensorBoard 日志），不指定则跳过训练曲线")
    p.add_argument("--out_dir", type=str, default=None,
                   help="图片输出目录，默认 results_dir/figures")
    return p.parse_args()


def _default_results_dir():
    return Path(__file__).resolve().parent / "evaluation_results"


def _default_model_dir():
    return Path(__file__).resolve().parent / "tinystories_model"


# ---------- 数据加载 ----------
def load_overall_summary(results_dir):
    p = Path(results_dir) / "overall_summary.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_persona_details(results_dir, persona, kind):
    """kind: 'persona_consistency' | 'task_quality'"""
    suffix = "persona_consistency_details.jsonl" if kind == "persona_consistency" else "task_quality_details.jsonl"
    p = Path(results_dir) / f"{persona}_{suffix}"
    if not p.exists():
        return []
    lines = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(json.loads(line))
    return lines


# ---------- 评估结果可视化 ----------
def plot_persona_consistency_radar(summary, out_path):
    """三人设 persona_consistency 雷达图（persona_match, naturalness, consistency）。"""
    if not summary:
        return
    personas = list(summary.keys())
    metrics = ["persona_match_avg", "naturalness_avg", "consistency_avg"]
    labels = ["Persona Match", "Naturalness", "Consistency"]
    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection="polar"))
    for i, persona in enumerate(personas):
        pc = summary[persona].get("persona_consistency") or {}
        vals = [pc.get(m, 0) for m in metrics]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, label=persona.capitalize())
        ax.fill(angles, vals, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 5)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    ax.set_title("Persona Consistency (Averages)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_task_quality_bars(summary, out_path):
    """三人设任务质量（helpfulness, fluency, correctness）柱状图。"""
    if not summary:
        return
    personas = list(summary.keys())
    metrics = ["helpfulness_avg", "fluency_avg", "correctness_avg"]
    labels = ["Helpfulness", "Fluency", "Correctness"]
    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, persona in enumerate(personas):
        tq = summary[persona].get("task_quality_preservation") or {}
        vals = [tq.get(m, 0) for m in metrics]
        ax.bar(x + i * w, vals, w, label=persona.capitalize())
    ax.set_xticks(x + w)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Average Score")
    ax.set_ylim(0, 5.5)
    ax.legend()
    ax.set_title("Task Quality Preservation by Persona")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_failure_distribution(summary, out_path):
    """各 persona 的 failure_type 堆叠柱状图。"""
    if not summary:
        return
    personas = list(summary.keys())
    failure_types = ["none", "too_neutral", "wrong_persona", "forced_style"]
    data = {}
    for ft in failure_types:
        data[ft] = []
        for p in personas:
            dist = (summary[p].get("persona_consistency") or {}).get("failure_distribution") or {}
            data[ft].append(dist.get(ft, 0))
    x = np.arange(len(personas))
    w = 0.6
    bottom = np.zeros(len(personas))
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#2ecc71", "#f39c12", "#e74c3c", "#3498db"]
    for ft, c in zip(failure_types, colors):
        ax.bar(x, data[ft], w, bottom=bottom, label=ft.replace("_", " ").title(), color=c)
        bottom += np.array(data[ft])
    ax.set_xticks(x)
    ax.set_xticklabels([p.capitalize() for p in personas])
    ax.set_ylabel("Count")
    ax.legend(loc="upper right")
    ax.set_title("Persona Consistency Failure Distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_error_distribution(summary, out_path):
    """各 persona 的 error_type 堆叠柱状图。"""
    if not summary:
        return
    personas = list(summary.keys())
    error_types = ["none", "incomplete", "off_topic", "vague"]
    data = {}
    for et in error_types:
        data[et] = []
        for p in personas:
            dist = (summary[p].get("task_quality_preservation") or {}).get("error_distribution") or {}
            data[et].append(dist.get(et, 0))
    x = np.arange(len(personas))
    w = 0.6
    bottom = np.zeros(len(personas))
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#2ecc71", "#f39c12", "#e74c3c", "#9b59b6"]
    for et, c in zip(error_types, colors):
        ax.bar(x, data[et], w, bottom=bottom, label=et.replace("_", " ").title(), color=c)
        bottom += np.array(data[et])
    ax.set_xticks(x)
    ax.set_xticklabels([p.capitalize() for p in personas])
    ax.set_ylabel("Count")
    ax.legend(loc="upper right")
    ax.set_title("Task Quality Error Distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_voice_characteristics(summary, out_path):
    """选取部分 voice 指标做三人设对比柱状图。"""
    if not summary:
        return
    personas = list(summary.keys())
    # 选通用且易解释的指标
    metrics = [
        "avg_tokens", "avg_sentences", "avg_sentence_length",
        "avg_type_token_ratio", "style_consistency_proxy_mean"
    ]
    labels = ["Avg Tokens", "Avg Sentences", "Avg Sent Len", "Type-Token Ratio", "Style Consistency"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    w = 0.25
    for i, persona in enumerate(personas):
        vc = summary[persona].get("voice_characteristics") or {}
        vals = [vc.get(m, 0) for m in metrics]
        ax.bar(x + i * w, vals, w, label=persona.capitalize())
    ax.set_xticks(x + w)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Value")
    ax.legend()
    ax.set_title("Voice Characteristics by Persona")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_single_voice_metric(summary, metric_key, title, ylabel, out_path):
    """对单一 voice 指标做三人设柱状图。"""
    if not summary:
        return
    personas = list(summary.keys())
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
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_ttr_and_bigram_metrics(summary, out_voice):
    """分别绘制 avg_type_token_ratio, corpus_type_token_ratio, avg_repeated_bigram_count 三人设对比图。"""
    if not summary:
        return
    metrics_config = [
        ("avg_type_token_ratio", "Avg Type-Token Ratio by Persona", "Avg Type-Token Ratio", "avg_type_token_ratio.png"),
        ("corpus_type_token_ratio", "Corpus Type-Token Ratio by Persona", "Corpus Type-Token Ratio", "corpus_type_token_ratio.png"),
        ("avg_repeated_bigram_count", "Avg Repeated Bigram Count by Persona", "Avg Repeated Bigram Count", "avg_repeated_bigram_count.png"),
    ]
    for metric_key, title, ylabel, fname in metrics_config:
        plot_single_voice_metric(summary, metric_key, title, ylabel, out_voice / fname)


def plot_category_breakdown_heatmap(summary, out_path, metric="helpfulness_avg"):
    """各 persona × 任务类别的 metric 热力图。"""
    if not summary:
        return
    personas = list(summary.keys())
    all_cats = set()
    for p in personas:
        cb = (summary[p].get("task_quality_preservation") or {}).get("category_breakdown") or {}
        all_cats.update(cb.keys())
    categories = sorted(all_cats)
    data = np.zeros((len(personas), len(categories)))
    for i, p in enumerate(personas):
        cb = (summary[p].get("task_quality_preservation") or {}).get("category_breakdown") or {}
        for j, c in enumerate(categories):
            data[i, j] = (cb.get(c) or {}).get(metric, np.nan)
    fig, ax = plt.subplots(figsize=(max(8, len(categories) * 0.5), 4))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=1, vmax=5)
    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(personas)))
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_yticklabels([p.capitalize() for p in personas])
    plt.colorbar(im, ax=ax, label=metric.replace("_", " ").title())
    ax.set_title(f"Task Quality by Category ({metric})")
    for i in range(len(personas)):
        for j in range(len(categories)):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_persona_scores_distribution(results_dir, out_dir):
    """从 JSONL 细节中绘制各 persona 的 persona_match / naturalness / consistency 分布（箱线图）。"""
    summary = load_overall_summary(results_dir)
    if not summary:
        return
    personas = list(summary.keys())
    metrics = ["persona_match", "naturalness", "consistency"]
    for metric in metrics:
        data_by_persona = []
        for p in personas:
            rows = load_persona_details(results_dir, p, "persona_consistency")
            vals = [r.get(metric) for r in rows if r.get(metric) is not None]
            data_by_persona.append(vals)
        if not any(data_by_persona):
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(data_by_persona, labels=[x.capitalize() for x in personas])
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"Distribution of {metric.replace('_', ' ').title()} by Persona")
        plt.tight_layout()
        out_path = Path(out_dir) / f"persona_{metric}_boxplot.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")


def plot_task_scores_distribution(results_dir, out_dir):
    """从 JSONL 细节中绘制各 persona 的 helpfulness / fluency / correctness 分布。"""
    summary = load_overall_summary(results_dir)
    if not summary:
        return
    personas = list(summary.keys())
    metrics = ["helpfulness", "fluency", "correctness"]
    for metric in metrics:
        data_by_persona = []
        for p in personas:
            rows = load_persona_details(results_dir, p, "task_quality")
            vals = [r.get(metric) for r in rows if r.get(metric) is not None]
            data_by_persona.append(vals)
        if not any(data_by_persona):
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(data_by_persona, labels=[x.capitalize() for x in personas])
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"Distribution of {metric.replace('_', ' ').title()} by Persona")
        plt.tight_layout()
        out_path = Path(out_dir) / f"task_{metric}_boxplot.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")


# ---------- tinystories_model 训练曲线 ----------
def load_tb_scalars(logdir, tags):
    """从 TensorBoard 日志目录读取 scalar 序列。返回 {tag: [(step, value), ...]}。"""
    if not HAS_TB:
        return {}
    logdir = Path(logdir)
    if not logdir.is_dir():
        return {}
    out = {t: [] for t in tags}
    try:
        ea = event_accumulator.EventAccumulator(str(logdir))
        ea.Reload()
        for tag in tags:
            events = ea.Scalars(tag)
            for e in events:
                out[tag].append((e.step, e.value))
    except Exception as e:
        print(f"TensorBoard 读取失败 ({logdir}): {e}")
        return {}
    return out


def _plot_single_training_curves(data, title_prefix, out_path):
    """Draw loss and perplexity into one figure and save. data: {tag: [(step, val), ...]}."""
    tags = ["Loss/train", "Loss/val", "Perplexity/train", "Perplexity/val"]
    if not any(data.get(t) for t in tags):
        return False
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if data.get("Loss/train"):
        steps, vals = zip(*data["Loss/train"])
        axes[0].plot(steps, vals, label="Train")
    if data.get("Loss/val"):
        steps, vals = zip(*data["Loss/val"])
        axes[0].plot(steps, vals, label="Val")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_title(f"{title_prefix} — Training & Validation Loss")
    if data.get("Perplexity/train"):
        steps, vals = zip(*data["Perplexity/train"])
        axes[1].plot(steps, vals, label="Train")
    if data.get("Perplexity/val"):
        steps, vals = zip(*data["Perplexity/val"])
        axes[1].plot(steps, vals, label="Val")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Perplexity")
    axes[1].legend()
    axes[1].set_title(f"{title_prefix} — Perplexity")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return True


def plot_training_curves(model_dir, out_dir):
    """绘制基础模型训练曲线（tinystories_model 根目录的 TensorBoard 日志）。"""
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        print(f"跳过训练曲线：目录不存在 {model_dir}")
        return
    tags = ["Loss/train", "Loss/val", "Perplexity/train", "Perplexity/val"]
    data = load_tb_scalars(model_dir, tags)
    if not any(data.get(t) for t in tags):
        print("未找到基础模型 TensorBoard 标量数据，跳过 tinystories_training_curves.png")
        return
    _plot_single_training_curves(data, "Base (TinyStories)", Path(out_dir) / "tinystories_training_curves.png")


# (name_for_title, subdir_or_none for base)
TRAINING_LOG_DIRS = [
    ("Base (TinyStories)", None),  # use model_dir root
    ("Friendly chat", "friendly_chat_model"),
    ("Robot chat", "robot_chat_model"),
    ("Sarcastic chat", "sarcastic_chat_model"),
]


def plot_all_training_curves(model_dir, out_dir):
    """为基础模型和三个人设 chat 模型分别绘制训练曲线，保存到 others/。"""
    model_dir = Path(model_dir)
    out_dir = Path(out_dir)
    if not model_dir.is_dir():
        return
    tags = ["Loss/train", "Loss/val", "Perplexity/train", "Perplexity/val"]
    for title_prefix, subdir in TRAINING_LOG_DIRS:
        logdir = model_dir if subdir is None else model_dir / subdir
        if not logdir.is_dir() and subdir is not None:
            continue
        data = load_tb_scalars(logdir, tags)
        if not data or not any(data.get(t) for t in tags):
            continue
        if subdir is None:
            fname = "base_model_training_curves.png"
        else:
            fname = subdir.replace("_chat_model", "") + "_chat_training_curves.png"
        if _plot_single_training_curves(data, title_prefix, out_dir / fname):
            pass  # already saved and printed


def collect_checkpoint_losses(logdir):
    """从目录下的 checkpoint-*.pth 中读取 (global_step, loss)。返回 [(step, loss), ...] 按 step 排序。"""
    if not HAS_TORCH:
        return []
    logdir = Path(logdir)
    if not logdir.is_dir():
        return []
    import re
    points = []
    for f in logdir.glob("checkpoint-*.pth"):
        m = re.match(r"checkpoint-(\d+)\.pth", f.name)
        if not m:
            continue
        step = int(m.group(1))
        try:
            ckpt = torch.load(f, map_location="cpu")
            loss = ckpt.get("loss")
            if loss is not None:
                points.append((step, float(loss)))
        except Exception:
            continue
    return sorted(points, key=lambda x: x[0])


def plot_checkpoint_loss_curves(model_dir, out_dir):
    """当无 TensorBoard 或作为补充：从 checkpoint-*.pth 绘制稀疏的「训练 loss / perplexity 随 step」图。"""
    model_dir = Path(model_dir)
    out_dir = Path(out_dir)
    if not model_dir.is_dir():
        return
    for title_prefix, subdir in TRAINING_LOG_DIRS:
        logdir = model_dir if subdir is None else model_dir / subdir
        if subdir is not None and not logdir.is_dir():
            continue
        points = collect_checkpoint_losses(logdir)
        if not points:
            continue
        steps, losses = zip(*points)
        ppl = [np.exp(l) for l in losses]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(steps, losses, "o-", markersize=4)
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Loss (at checkpoint)")
        axes[0].set_title(f"{title_prefix} — Training loss at checkpoints")
        axes[1].plot(steps, ppl, "o-", markersize=4)
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Perplexity (at checkpoint)")
        axes[1].set_title(f"{title_prefix} — Perplexity at checkpoints")
        plt.tight_layout()
        if subdir is None:
            fname = "base_model_checkpoint_loss.png"
        else:
            fname = subdir.replace("_chat_model", "") + "_chat_checkpoint_loss.png"
        out_path = out_dir / fname
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")


def plot_model_config_if_exists(model_dir, out_dir):
    """若存在 args.json，可生成模型配置概览图（简单文本/表格）。"""
    args_path = Path(model_dir) / "args.json"
    if not args_path.exists():
        return
    with open(args_path, "r", encoding="utf-8") as f:
        args = json.load(f)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    rows = [f"{k}: {v}" for k, v in sorted(args.items())]
    ax.text(0.05, 0.95, "TinyStories Model Config (args.json)", fontsize=14, fontweight="bold", transform=ax.transAxes)
    ax.text(0.05, 0.85, "\n".join(rows), fontsize=10, transform=ax.transAxes, verticalalignment="top", family="monospace")
    out_path = Path(out_dir) / "tinystories_model_config.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------- Subfolders ----------
SUBDIRS = {
    "persona_consistency": "Persona Consistency",
    "voice_characteristics": "Voice Characteristics",
    "task_quality_preservation": "Task Quality Preservation",
    "others": "Others",
}

# English one-line summaries for each figure (folder -> filename -> summary)
FIGURE_SUMMARIES = {
    "persona_consistency": {
        "persona_consistency_radar.png": "Radar chart of average persona match, naturalness, and consistency scores across the three personas.",
        "failure_distribution.png": "Stacked bar chart of persona consistency failure types (none, too neutral, wrong persona, forced style) per persona.",
        "persona_persona_match_boxplot.png": "Boxplot of persona match scores by persona, showing distribution and outliers.",
        "persona_naturalness_boxplot.png": "Boxplot of naturalness scores by persona.",
        "persona_consistency_boxplot.png": "Boxplot of consistency scores by persona.",
    },
    "voice_characteristics": {
        "voice_characteristics.png": "Bar chart comparing voice metrics (avg tokens, sentences, sentence length, type-token ratio, style consistency) across personas.",
        "avg_type_token_ratio.png": "Bar chart of average type-token ratio by persona (friendly, robot, sarcastic).",
        "corpus_type_token_ratio.png": "Bar chart of corpus type-token ratio by persona.",
        "avg_repeated_bigram_count.png": "Bar chart of average repeated bigram count by persona.",
    },
    "task_quality_preservation": {
        "task_quality_bars.png": "Bar chart of average helpfulness, fluency, and correctness scores by persona.",
        "error_distribution.png": "Stacked bar chart of task quality error types (none, incomplete, off topic, vague) per persona.",
        "category_helpfulness_heatmap.png": "Heatmap of average helpfulness by persona and task category.",
        "category_correctness_heatmap.png": "Heatmap of average correctness by persona and task category.",
        "task_helpfulness_boxplot.png": "Boxplot of helpfulness scores by persona.",
        "task_fluency_boxplot.png": "Boxplot of fluency scores by persona.",
        "task_correctness_boxplot.png": "Boxplot of correctness scores by persona.",
    },
    "others": {
        "tinystories_training_curves.png": "Training and validation loss and perplexity curves from base TinyStories model training (TensorBoard).",
        "base_model_training_curves.png": "Same as above: base (pretrained) TinyStories model training curves.",
        "friendly_chat_training_curves.png": "Training and validation loss and perplexity for the Friendly persona chat model (instruction tuning).",
        "robot_chat_training_curves.png": "Training and validation loss and perplexity for the Robot persona chat model (instruction tuning).",
        "sarcastic_chat_training_curves.png": "Training and validation loss and perplexity for the Sarcastic persona chat model (instruction tuning).",
        "base_model_checkpoint_loss.png": "Sparse training loss and perplexity at each saved checkpoint (from checkpoint-*.pth); no validation curve.",
        "friendly_chat_checkpoint_loss.png": "Sparse loss/perplexity at checkpoints for Friendly chat model (if checkpoint-*.pth exist).",
        "robot_chat_checkpoint_loss.png": "Sparse loss/perplexity at checkpoints for Robot chat model (if checkpoint-*.pth exist).",
        "sarcastic_chat_checkpoint_loss.png": "Sparse loss/perplexity at checkpoints for Sarcastic chat model (if checkpoint-*.pth exist).",
        "tinystories_model_config.png": "Overview of TinyStories model training arguments (args.json).",
    },
}


def write_figure_summaries(out_dir):
    """Write FIGURE_SUMMARIES.md with a short English summary for each image."""
    out_dir = Path(out_dir)
    lines = [
        "# Figure Summaries (Evaluation & Model)",
        "",
        "Short English description for each generated figure, grouped by folder.",
        "",
    ]
    for folder, title in SUBDIRS.items():
        lines.append(f"## {title} (`{folder}/`)")
        lines.append("")
        summaries = FIGURE_SUMMARIES.get(folder, {})
        folder_path = out_dir / folder
        if not folder_path.is_dir():
            continue
        for fname in sorted(folder_path.iterdir()):
            if fname.suffix.lower() not in (".png", ".jpg", ".jpeg", ".pdf"):
                continue
            name = fname.name
            summary = summaries.get(name, "No summary.")
            lines.append(f"- **{name}** — {summary}")
        lines.append("")
    path = out_dir / "FIGURE_SUMMARIES.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {path}")


# ---------- Main ----------
def main():
    args = parse_args()
    results_dir = Path(args.results_dir or _default_results_dir())
    model_dir = Path(args.model_dir or _default_model_dir())
    out_dir = Path(args.out_dir or results_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create four subfolders
    out_persona = out_dir / "persona_consistency"
    out_voice = out_dir / "voice_characteristics"
    out_task = out_dir / "task_quality_preservation"
    out_others = out_dir / "others"
    for d in (out_persona, out_voice, out_task, out_others):
        d.mkdir(parents=True, exist_ok=True)

    summary = load_overall_summary(results_dir)
    if not summary:
        print(f"未找到 overall_summary.json，请指定正确的 --results_dir。当前: {results_dir}")
        return

    # Persona consistency
    plot_persona_consistency_radar(summary, out_persona / "persona_consistency_radar.png")
    plot_failure_distribution(summary, out_persona / "failure_distribution.png")
    plot_persona_scores_distribution(results_dir, out_persona)

    # Voice characteristics
    plot_voice_characteristics(summary, out_voice / "voice_characteristics.png")
    plot_ttr_and_bigram_metrics(summary, out_voice)

    # Task quality preservation
    plot_task_quality_bars(summary, out_task / "task_quality_bars.png")
    plot_error_distribution(summary, out_task / "error_distribution.png")
    plot_category_breakdown_heatmap(summary, out_task / "category_helpfulness_heatmap.png", "helpfulness_avg")
    plot_category_breakdown_heatmap(summary, out_task / "category_correctness_heatmap.png", "correctness_avg")
    plot_task_scores_distribution(results_dir, out_task)

    # Others (model-related): base + three persona chat training curves
    plot_training_curves(model_dir, out_others)
    plot_all_training_curves(model_dir, out_others)
    # Sparse loss from checkpoint-*.pth (when TensorBoard missing or as supplement)
    plot_checkpoint_loss_curves(model_dir, out_others)
    plot_model_config_if_exists(model_dir, out_others)

    # Write English summaries for each figure
    write_figure_summaries(out_dir)
    print(f"所有图片已保存到: {out_dir}")


if __name__ == "__main__":
    main()
