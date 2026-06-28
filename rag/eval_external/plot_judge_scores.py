"""
docs/evaluation_v1_judge_scores.json을 읽어 모델별 평균 점수를 계산하고
막대그래프(docs/evaluation_v1_scores.png)를 생성한다.

실행: python3 rag/eval_external/plot_judge_scores.py
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
SCORES_PATH = os.path.join(PROJECT_ROOT, "docs", "evaluation_v1_judge_scores.json")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "docs", "evaluation_v1_scores.png")

MODELS = ["ella", "gemma", "gemini"]
MODEL_LABELS = {"ella": "Ella (23M)", "gemma": "Gemma (8B)", "gemini": "Gemini 2.5 Flash"}
CRITERIA = ["정확성", "관련성", "자연스러움"]
CRITERIA_LABELS = {"정확성": "Accuracy", "관련성": "Relevance", "자연스러움": "Fluency"}


def load_scores():
    with open(SCORES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_averages(data):
    """모델별, 기준별 평균과 전체 평균을 계산."""
    totals = {m: {c: 0 for c in CRITERIA} for m in MODELS}
    n = len(data)

    for item in data:
        for m in MODELS:
            for c in CRITERIA:
                totals[m][c] += item["scores"][m][c]

    averages = {m: {c: round(totals[m][c] / n, 2) for c in CRITERIA} for m in MODELS}
    overall = {m: round(sum(averages[m].values()) / len(CRITERIA), 2) for m in MODELS}
    return averages, overall


def print_table(averages, overall):
    print(f"{'Model':<20}" + "".join(f"{CRITERIA_LABELS[c]:<12}" for c in CRITERIA) + "Overall")
    for m in MODELS:
        row = f"{MODEL_LABELS[m]:<20}"
        row += "".join(f"{averages[m][c]:<12}" for c in CRITERIA)
        row += f"{overall[m]}"
        print(row)


def plot_bar_chart(averages, overall):
    x_labels = [CRITERIA_LABELS[c] for c in CRITERIA] + ["Overall"]
    colors = {"ella": "#C44E52", "gemma": "#4C72B0", "gemini": "#55A868"}

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_width = 0.25
    x = range(len(x_labels))

    for i, m in enumerate(MODELS):
        values = [averages[m][c] for c in CRITERIA] + [overall[m]]
        positions = [xi + (i - 1) * bar_width for xi in x]
        ax.bar(positions, values, width=bar_width, label=MODEL_LABELS[m], color=colors[m])

    ax.set_xticks(list(x))
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Score (0-5)")
    ax.set_ylim(0, 5.5)
    ax.set_title("Evaluation v1: Ella vs Gemma vs Gemini (Judge: Gemini 2.5 Flash)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"\n그래프 저장: {OUTPUT_PATH}")


if __name__ == "__main__":
    data = load_scores()
    averages, overall = compute_averages(data)
    print_table(averages, overall)
    plot_bar_chart(averages, overall)