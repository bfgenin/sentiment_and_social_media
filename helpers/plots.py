# helpers/plots.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib_venn import venn3

def plot_metric_bar(results_df, metric, title_suffix=None):
    """
    Generic horizontal bar plot for any metric in results_df.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain columns "Model" and the metric name.
    metric : str
        Column name of the metric to plot, e.g. "F1_macro", "Accuracy".
    title_suffix : str
        Label for the x-axis and title; if None, metric is used.
    """
    if results_df is None or results_df.empty:
        print("results_df is empty or None — no plot generated.")
        return

    if metric not in results_df.columns:
        print(f"Column '{metric}' not found in results_df — no plot generated.")
        return

    if title_suffix is None:
        title_suffix = metric

    # Sort by chosen metric
    plot_df = results_df.sort_values(metric, ascending=False).copy()
    plot_df["Model"] = plot_df["Model"].astype(str)

    # Color rules
    colors = []
    for m in plot_df["Model"]:
        m_lower = m.lower()
        if "two-stage" in m_lower:
            colors.append("green")
        elif "clean" in m_lower:
            colors.append("blue")
        else:
            colors.append("gray")

    y_pos = np.arange(len(plot_df))

    plt.figure(figsize=(12, 7))
    bars = plt.barh(y_pos, plot_df[metric], color=colors)

    plt.yticks(y_pos, plot_df["Model"])
    plt.gca().invert_yaxis()

    plt.title(f"Model Comparison: {title_suffix}", fontsize=16)
    plt.xlabel(title_suffix, fontsize=14)
    plt.ylabel("Model", fontsize=14)

    # Add metric value at bar end
    for bar, val in zip(bars, plot_df[metric]):
        plt.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center"
        )

    # Legend
    legend_handles = [
        Patch(facecolor="gray",   label="Full-data models"),
        Patch(facecolor="blue",   label="Clean-label models"),
        Patch(facecolor="green",  label="Two-stage models"),
    ]
    plt.legend(handles=legend_handles, title="Model Type", loc="lower right")

    plt.tight_layout()
    plt.show()


# Wrapper functions for specific metrics

def plot_macro_f1(results_df):
    plot_metric_bar(results_df, metric="F1_macro", title_suffix="Macro-F1")

def plot_accuracy(results_df):
    plot_metric_bar(results_df, metric="Accuracy", title_suffix="Accuracy")

def plot_recall_macro(results_df):
    plot_metric_bar(results_df, metric="Recall_macro", title_suffix="Macro Recall")

def plot_precision_macro(results_df):
    plot_metric_bar(results_df, metric="Precision_macro", title_suffix="Macro Precision")

def infer_labels_from_cm(cm):
    """Infer label names from confusion matrix size."""
    n = cm.shape[0]
    if n == 2:
        return ["negative", "positive"]
    elif n == 3:
        return ["negative", "neutral", "positive"]
    else:
        return [f"class {i}" for i in range(n)]
    
def plot_per_class_scores(confusion_matrices, models_to_compare):
    """
    Compute and plot per-class F1/precision/recall for selected models.
    
    Parameters
    ----------
    confusion_matrices : dict
        Dict mapping model_name → confusion_matrix (numpy array)
    
    models_to_compare : list of str
        List of model names to include
    
    Returns
    -------
    per_class_df : pd.DataFrame
        Table of per-class precision/recall/F1 for all selected models
    """

    rows = []

    for model_name in models_to_compare:
        if model_name not in confusion_matrices:
            print(f"Skipping {model_name}: no stored confusion matrix.")
            continue
        
        cm = confusion_matrices[model_name]
        labels = infer_labels_from_cm(cm)

        for i, cls in enumerate(labels):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1        = (2 * precision * recall / (precision + recall)
                         if precision + recall > 0 else 0.0)

            rows.append({
                "Model": model_name,
                "Class": cls,
                "Precision_class": precision,
                "Recall_class": recall,
                "F1_class": f1
            })

    per_class_df = pd.DataFrame(rows)

    # --- Plot ---
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=per_class_df,
        x="F1_class",
        y="Model",
        hue="Class"
    )
    plt.title("Per-Class F1 Comparison Across Models")
    plt.xlabel("F1 Score")
    plt.ylabel("Model")
    plt.legend(title="Class")
    plt.tight_layout()
    plt.show()

    return per_class_df

def plot_cm_diff(confusion_matrices, model_A, model_B):
    """
    Plot the difference between two confusion matrices: cm_B - cm_A.

    Parameters
    ----------
    confusion_matrices : dict
        Dict mapping model_name -> confusion_matrix (numpy array)

    model_A : str
        Name of the baseline model (subtrahend).

    model_B : str
        Name of the comparison model (minuend).

    Notes
    -----
    Positive cells in the heatmap mean model_B has *more* of that
    (true, pred) count than model_A; negative means fewer.
    """

    if model_A not in confusion_matrices:
        print(f"{model_A} not found in confusion_matrices")
        return
    if model_B not in confusion_matrices:
        print(f"{model_B} not found in confusion_matrices")
        return

    cm_A = confusion_matrices[model_A]
    cm_B = confusion_matrices[model_B]

    # Shape check
    if cm_A.shape != cm_B.shape:
        raise ValueError(
            f"Confusion matrices have different shapes: {cm_A.shape} vs {cm_B.shape}"
        )

    labels = infer_labels_from_cm(cm_A)
    cm_diff = cm_B - cm_A  # positive = B has more of that cell

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_diff,
        annot=True,
        fmt="+d",
        cmap="bwr",
        center=0,
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title(f"Confusion Matrix Difference: {model_B} - {model_A}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


def venn_top_words_from_extracted(platform_top_words, *, stage="stage1", cls="neutral",
                                  platforms=("reddit", "koo", "bluesky")):
    """
    platform_top_words: output of extract_two_stage_top_words_by_platform()
    stage: "stage1" or "stage2"
    cls: class name inside that stage (stage1: neutral/nonneutral; stage2: positive/negative)
    platforms: exactly 3 platform keys for venn3
    """
    if len(platforms) != 3:
        raise ValueError("venn3 requires exactly 3 platforms")

    sets = []
    for p in platforms:
        if p not in platform_top_words:
            raise KeyError(f"Platform '{p}' not found. Available: {list(platform_top_words.keys())}")

        stage_dict = platform_top_words[p].get(stage)
        if stage_dict is None:
            raise ValueError(f"Platform '{p}' has no '{stage}' data")

        words = stage_dict.get(cls)
        if words is None:
            raise ValueError(
                f"Class '{cls}' not found for {stage} on platform '{p}'. "
                f"Available: {list(stage_dict.keys())}"
            )

        sets.append(set(words))

    A, B, C = sets
    plt.figure(figsize=(8, 8))
    venn3([A, B, C], set_labels=(platforms[0].title(), platforms[1].title(), platforms[2].title()))
    plt.title(f"Overlap of Top Words — {stage} class '{cls}'")
    plt.show()
