import json
import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from alla.utils import PROJECT_ROOT


def exp1():
    # Initialize seaborn
    sns.set_theme(style="darkgrid")
    sns.set_palette("Set2")
    sns.set_context("notebook")

    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results/exp-1/llama3-70b")
    save_path1 = os.path.join(RESULTS_DIR, "figs/cat-dist.png")
    save_path2 = os.path.join(RESULTS_DIR, "figs/growth-dist.png")

    # Load ALLA results
    alla_results = json.load(open(os.path.join(RESULTS_DIR, "alla_results.json"), 'r'))
    # For now we use the .5 percentile for plotting
    alla_data = [
        {
            "category": max(asset['category'], key=lambda x: asset['category'][x]),
            "growth": asset['growth']["0.5"],
        }
        for asset in alla_results.values()
    ]

    llm_results: dict = json.load(open(os.path.join(RESULTS_DIR, "llm_results.json"), 'r'))
    llm_data = [
        {
            "category": v["category"],
            "growth": v["growth"],
        }
        for v in llm_results.values()
    ]
    truth_data = [
        {
            "category": v["truth"]["category"],
            "growth": v["truth"]["growth"],
        }
        for v in llm_results.values()
    ]
    # Create a combined pandas df of the category and growth values for llm and truth sets
    df = pd.concat(
        [
            pd.DataFrame(alla_data).assign(source="ALLA"),
            pd.DataFrame(llm_data).assign(source="LLM"),
            pd.DataFrame(truth_data).assign(source="Truth")
        ], ignore_index=True
    )

    # <><>< Figure 1 ><><>
    # Category count plot
    plt.figure(figsize=(4, 4))
    sns.countplot(
        data=df, x="category", hue="source"
    )
    # Remove labels as it is self-evident
    plt.xlabel("")
    plt.xticks(ticks=[0, 1, 2], labels=["High Growth", "Stable", "Time-bomb"])
    plt.ylabel("")
    plt.legend(title=None)
    plt.tight_layout()
    # Save and close
    print(f"Saving to {save_path1}")
    plt.savefig(save_path1)
    plt.close()

    # <><>< Figure 2 ><><>
    # We plot the growth distributions for each category, and compare the sources
    g = sns.FacetGrid(
        df,
        col="category",
        hue="source",
        sharex=True,
        sharey=True,
        height=4,
        aspect=1.2,
    )

    # Map the growth distribution (using histplot or kdeplot)
    g.map(sns.histplot, "growth", alpha=0.5, kde=False, bins=20)
    g.add_legend()
    # Improve the titles
    for ax, title in zip(g.axes.flat, ["High Growth", "Stable", "Time-bomb"]):
        ax.set_title(title)
        ax.set_xlabel("Growth")

    print(f"Saving to {save_path2}")
    plt.savefig(save_path2)
    plt.close()
    # <><>< Confusion Matrix ><><>

    # Create a confusion matrix for the category predictions
    y_true = df[df["source"] == "Truth"]["category"]
    y_pred = df[df["source"] == "LLM"]["category"]

    cm = confusion_matrix(y_true, y_pred, labels=["High Growth", "Stable", "Time-bomb"])
    cm_df = pd.DataFrame(cm, index=["High Growth", "Stable", "Time-bomb"], columns=["High Growth", "Stable", "Time-bomb"])

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    save_path3 = os.path.join(RESULTS_DIR, "figs/confusion_matrix_llm.png")
    print(f"Saving to {save_path3}")
    plt.savefig(save_path3)
    plt.close()

    # <><>< Confusion Matrix for ALLA ><><>

    # Create a confusion matrix for the category predictions for ALLA results
    y_true_alla = df[df["source"] == "Truth"]["category"]
    y_pred_alla = df[df["source"] == "ALLA"]["category"]

    cm_alla = confusion_matrix(y_true_alla, y_pred_alla, labels=["High Growth", "Stable", "Time-bomb"])
    cm_alla_df = pd.DataFrame(cm_alla, index=["High Growth", "Stable", "Time-bomb"], columns=["High Growth", "Stable", "Time-bomb"])

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_alla_df, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix for ALLA")
    plt.tight_layout()

    save_path4 = os.path.join(RESULTS_DIR, "figs/confusion_matrix_alla.png")
    print(f"Saving to {save_path4}")
    plt.savefig(save_path4)
    plt.close()

    # Classification report for LLM results
    report_llm = classification_report(y_true, y_pred, target_names=["High Growth", "Stable", "Time-bomb"], output_dict=True)
    report_llm_df = pd.DataFrame(report_llm).transpose()

    save_path_llm_report = os.path.join(RESULTS_DIR, "figs/classification_report_llm.tex")
    print(f"Saving LLM classification report to {save_path_llm_report}")
    with open(save_path_llm_report, 'w') as f:
        f.write(report_llm_df.to_latex())

    # Classification report for ALLA results
    report_alla = classification_report(y_true_alla, y_pred_alla, target_names=["High Growth", "Stable", "Time-bomb"], output_dict=True)
    report_alla_df = pd.DataFrame(report_alla).transpose()

    save_path_alla_report = os.path.join(RESULTS_DIR, "figs/classification_report_alla.tex")
    print(f"Saving ALLA classification report to {save_path_alla_report}")
    with open(save_path_alla_report, 'w') as f:
        f.write(report_alla_df.to_latex())


if __name__ == '__main__':
    # Accept first argument as function name to be called
    import sys
    from sklearn.metrics import confusion_matrix
    import numpy as np
    import seaborn as sns
    from sklearn.metrics import classification_report

    if len(sys.argv) < 2:
        raise ValueError("Please provide a function name to call")

    func_name = sys.argv[1]
    if func_name not in globals():
        raise ValueError(f"Function {func_name} not found")

    print(f"Calling {func_name}")
    globals()[func_name]()
