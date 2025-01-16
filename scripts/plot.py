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

    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results/exp-1/test")
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


if __name__ == '__main__':
    # Accept first argument as function name to be called
    import sys

    if len(sys.argv) < 2:
        raise ValueError("Please provide a function name to call")

    func_name = sys.argv[1]
    if func_name not in globals():
        raise ValueError(f"Function {func_name} not found")

    print(f"Calling {func_name}")
    globals()[func_name]()
