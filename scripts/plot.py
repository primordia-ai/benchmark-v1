import json
import os

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import matthews_corrcoef

from cristal.env import PROJECT_ROOT


def exp1(results_name: str):
    # ================
    # Some setup

    # Initialize seaborn
    sns.set_theme(style="darkgrid")
    sns.set_palette("Set2")
    sns.set_context("notebook")

    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", 'exp-1', results_name)
    SAVE_DIR = os.path.join(RESULTS_DIR, "figs")
    save_fmt = "pdf"

    # Create figs directory
    os.makedirs(SAVE_DIR, exist_ok=True)

    lbl_map = {
        "stable": "Stable",
        "high_growth": "High Growth",
        "time_bomb": "Time-bomb",
        "aggregate": "Aggregate"
    }

    # ================
    # Load data

    try:
        cristal_results = json.load(open(os.path.join(RESULTS_DIR, "cristal_results.json"), 'r'))
    except FileNotFoundError:
        # Older format
        print("cristal_results.json not found, trying older format...")
        cristal_results = json.load(open(os.path.join(RESULTS_DIR, "alla_results.json"), 'r'))

    llm_results: dict = json.load(open(os.path.join(RESULTS_DIR, "llm_results.json"), 'r'))

    # Combine the set of errored assets from both CRISTAL and LLM, to keep things balanced
    errors = set()
    errors |= {a_id for a_id, asset in cristal_results.items() if "error" in asset}
    errors |= {a_id for a_id, asset in llm_results.items() if "error" in asset}

    # Load CRISTAL results -- filter out errors
    cristal_data = [
        {
            "asset": a_id,
            "category": max(asset['category'], key=lambda x: asset['category'][x]),
            "true_category": asset['truth']['category'],
            "growth": asset['growth'][2][1],  # p_50
            "true_growth": asset['truth']['growth'],
            "class_loss": asset['class_loss'],
            "parametric_score": asset['parametric_score'],
            "non_parametric_loss": asset['non_parametric_loss']
        }
        for a_id, asset in cristal_results.items() if a_id not in errors
    ]
    # LLM data
    llm_data = [
        {
            "asset": a_id,
            "category": max(asset['category'], key=lambda x: asset['category'][x]),
            "true_category": asset['truth']['category'],
            "growth": asset['growth'][2][1],  # p_50
            "true_growth": asset['truth']['growth'],
            "class_loss": asset['class_loss'],
            "parametric_score": asset['parametric_score'],
            "non_parametric_loss": asset['non_parametric_loss']
        }
        for a_id, asset in llm_results.items() if a_id not in errors
    ]
    truth_data = [
        {
            "asset": a_id,
            "category": v["truth"]["category"],
            "growth": v["truth"]["growth"],
        }
        for a_id, v in llm_results.items() if a_id not in errors
    ]

    # Create a combined pandas df of the category and growth values for llm and truth sets
    df = pd.concat(
        [
            pd.DataFrame(cristal_data).assign(source="CRISTAL"),
            pd.DataFrame(llm_data).assign(source="LLM"),
            pd.DataFrame(truth_data).assign(source="Truth")
        ], ignore_index=True
    )

    # Often we want to filter out the Truth data
    df_nt = df[df["source"] != "Truth"]

    # We can also create an aggregate category that duplicates the data but is helpful
    # for plotting alongside the individual categories
    df_aggregate = df_nt.copy()
    df_aggregate["true_category"] = "aggregate"
    df_aggregate["category"] = "aggregate"

    # ================
    # On to figures

    # <><>< 1. Category Count ><><>
    save_path = os.path.join(SAVE_DIR, f"cat-dist.{save_fmt}")
    plt.figure(figsize=(4, 4))
    sns.countplot(
        data=df, x="category", hue="source", order=["high_growth", "stable", "time_bomb"]
    )
    # Remove x label
    plt.xlabel("")
    # Reformat tick labels using lbl_map
    labels = [lbl_map[l.get_text()] for l in plt.gca().get_xticklabels()]
    plt.xticks(ticks=[0, 1, 2], labels=labels)
    plt.ylabel("")
    plt.legend(title=None)
    plt.tight_layout()
    # Save and close
    print(f"Saving to {save_path}")
    plt.savefig(save_path)
    plt.close()

    # <><>< 2. Growth Distribution ><><>
    # We plot the growth distributions for each category, and compare the sources
    save_path = os.path.join(SAVE_DIR, f"growth-dist.{save_fmt}")

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

    print(f"Saving to {save_path}")
    plt.savefig(save_path)
    plt.close()

    # <><>< 3. Confusion Matrix, LLM ><><>
    # Create a confusion matrix for the category predictions for LLM vs Truth sources
    save_path = os.path.join(SAVE_DIR, f"confusion-llm.{save_fmt}")

    # We need to sort by 'asset' column
    y_true = df[df["source"] == "Truth"].sort_values("asset")["category"]
    y_pred = df[df["source"] == "LLM"].sort_values("asset")["category"]

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm, index=["High Growth", "Stable", "Time-bomb"], columns=["High Growth", "Stable", "Time-bomb"]
    )

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    print(f"Saving to {save_path}")
    plt.savefig(save_path)
    plt.close()

    ### Latex export ###
    # Classification report for LLM results
    report_llm = classification_report(
        y_true, y_pred, target_names=["High Growth", "Stable", "Time-bomb"], output_dict=True
    )
    report_llm_df = pd.DataFrame(report_llm).transpose()
    # Calculate accuracy and MCC
    accuracy_llm = (y_true.reset_index(drop=True) == y_pred.reset_index(drop=True)).mean()
    mcc_llm = matthews_corrcoef(y_true, y_pred)
    # Save the updated report to LaTeX
    save_path_llm_report = os.path.join(RESULTS_DIR, "figs/classification_report_llm.tex")
    caption = f"Classification report for LLM results. Accuracy: {accuracy_llm:.2f}, MCC: {mcc_llm:.2f}"
    print(f"Saving LLM classification report to {save_path_llm_report}")
    with open(save_path_llm_report, 'w') as f:
        f.write(report_llm_df.to_latex(caption=caption, label="tab:classification_report_llm"))

    # <><>< 4. Confusion Matrix, CRISTAL ><><>
    save_path = os.path.join(SAVE_DIR, f"confusion-cristal.{save_fmt}")

    y_true_cristal = df[df["source"] == "Truth"].sort_values("asset")["category"]
    y_pred_cristal = df[df["source"] == "CRISTAL"].sort_values("asset")["category"]

    cm_cristal = confusion_matrix(y_true_cristal, y_pred_cristal)
    cm_cristal_df = pd.DataFrame(
        cm_cristal, index=["High Growth", "Stable", "Time-bomb"], columns=["High Growth", "Stable", "Time-bomb"]
    )

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_cristal_df, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix for CRISTAL")
    plt.tight_layout()

    print(f"Saving to {save_path}")
    plt.savefig(save_path)
    plt.close()

    ### Latex export ###
    report_cristal = classification_report(
        y_true_cristal, y_pred_cristal, target_names=["High Growth", "Stable", "Time-bomb"], output_dict=True
    )
    report_cristal_df = pd.DataFrame(report_cristal).transpose()
    # Calculate accuracy and MCC
    accuracy_cristal = (y_true_cristal.reset_index(drop=True) == y_pred_cristal.reset_index(drop=True)).mean()
    mcc_cristal = matthews_corrcoef(y_true_cristal, y_pred_cristal)

    # Save the updated report to LaTeX with caption
    save_path_cristal_report = os.path.join(RESULTS_DIR, "figs/classification_report_cristal.tex")
    caption = f"Classification report for CRISTAL results. Accuracy: {accuracy_cristal:.2f}, MCC: {mcc_cristal:.2f}"
    print(f"Saving CRISTAL classification report to {save_path_cristal_report}")
    with open(save_path_cristal_report, 'w') as f:
        f.write(report_cristal_df.to_latex(caption=caption, label="tab:classification_report_cristal"))

    # <><>< 5. Class Loss Boxplot ><><>
    # Create a boxplot of the class loss for each source (except for Truth), grouped by the corresponding
    # true category. Includes "Aggregate" group as well
    save_path = os.path.join(SAVE_DIR, f"class-loss-box.{save_fmt}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4), gridspec_kw={'width_ratios': [3, 1]})

    # --- Main categories ---
    sns.boxplot(
        data=df_nt, x="true_category", y="class_loss", hue="source",
        log_scale=True, showfliers=False, ax=ax1
    )
    sns.stripplot(
        data=df_nt, x="true_category", y="class_loss", hue="source", size=3.2,
        dodge=True, jitter=True, edgecolor="auto", linewidth=1,
        legend=False, ax=ax1
    )
    ax1.set_xlabel("")
    ax1.set_ylabel("Class Loss")
    ax1.legend(title=None)
    # Reformat tick labels using lbl_map
    labels = [lbl_map[l.get_text()] for l in ax1.get_xticklabels()]
    ax1.set_xticks(ticks=[0, 1, 2], labels=labels)

    # --- Aggregate category ---
    sns.boxplot(
        data=df_aggregate, x="true_category", y="class_loss", hue="source",
        log_scale=True, showfliers=False, ax=ax2
    )
    sns.stripplot(
        data=df_aggregate, x="true_category", y="class_loss", hue="source", size=3.2,
        dodge=True, jitter=True, edgecolor="auto", linewidth=1,
        legend=False, ax=ax2
    )
    ax2.set_xlabel("")
    ax2.set_ylabel("")  # Remove y-label
    ax2.set_yticklabels([])  # Remove y-ticks
    ax2.set_xticks(ticks=[0], labels=["Aggregate"])
    ax2.get_legend().remove()
    # Align y-axis limits between both plots
    ax2.set_ylim(ax1.get_ylim())

    # Improve layout
    plt.tight_layout()

    print(f"Saving to {save_path}")
    plt.savefig(save_path)
    plt.close()

    # <><>< 6. Parametric Score Boxplot ><><>
    # Similar to above
    save_path = os.path.join(SAVE_DIR, f"parametric-score-box.{save_fmt}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4), gridspec_kw={'width_ratios': [3, 1]})

    # --- Main categories ---
    sns.boxplot(
        data=df_nt, x="true_category", y="parametric_score", hue="source",
        showfliers=False, ax=ax1
    )
    sns.stripplot(
        data=df_nt, x="true_category", y="parametric_score", hue="source", size=3.2,
        dodge=True, jitter=True, edgecolor="auto", linewidth=1,
        legend=False, ax=ax1
    )
    ax1.set_xlabel("")
    ax1.set_ylabel("Parametric Score")
    # Reformat tick labels using lbl_map
    labels = [lbl_map[l.get_text()] for l in ax1.get_xticklabels()]
    ax1.set_xticks(ticks=[0, 1, 2], labels=labels)
    ax1.legend(title=None)

    # --- Aggregate category ---
    sns.boxplot(
        data=df_aggregate, x="true_category", y="parametric_score", hue="source",
        showfliers=False, ax=ax2
    )
    sns.stripplot(
        data=df_aggregate, x="true_category", y="parametric_score", hue="source", size=3.2,
        dodge=True, jitter=True, edgecolor="auto", linewidth=1,
        legend=False, ax=ax2
    )
    ax2.set_xlabel("")
    ax2.set_ylabel("")  # Remove y-label
    ax2.set_yticklabels([])  # Remove y-ticks
    ax2.set_xticks(ticks=[0], labels=["Aggregate"])
    ax2.get_legend().remove()
    # Align y-axis limits between both plots
    ax2.set_ylim(ax1.get_ylim())

    # Improve layout
    plt.tight_layout()

    print(f"Saving to {save_path}")
    plt.savefig(save_path)
    plt.close()

    # <><>< 7. Non-Parametric Loss Boxplot ><><>
    # Similar to above
    save_path = os.path.join(SAVE_DIR, f"non-parametric-loss-box.{save_fmt}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4), gridspec_kw={'width_ratios': [3, 1]})

    # --- Main categories ---
    sns.boxplot(
        data=df_nt, x="true_category", y="non_parametric_loss", hue="source",
        showfliers=False, ax=ax1
    )
    sns.stripplot(
        data=df_nt, x="true_category", y="non_parametric_loss", hue="source", size=3.2,
        dodge=True, jitter=True, edgecolor="auto", linewidth=1,
        legend=False, ax=ax1
    )
    ax1.set_xlabel("")
    ax1.set_ylabel("Non-Parametric Loss")
    # Reformat tick labels using lbl_map
    labels = [lbl_map[l.get_text()] for l in ax1.get_xticklabels()]
    ax1.set_xticks(ticks=[0, 1, 2], labels=labels)
    ax1.legend(title=None)

    # --- Aggregate category ---
    sns.boxplot(
        data=df_aggregate, x="true_category", y="non_parametric_loss", hue="source",
        showfliers=False, ax=ax2
    )
    sns.stripplot(
        data=df_aggregate, x="true_category", y="non_parametric_loss", hue="source", size=3.2,
        dodge=True, jitter=True, edgecolor="auto", linewidth=1,
        legend=False, ax=ax2
    )
    ax2.set_xlabel("")
    ax2.set_ylabel("")  # Remove y-label
    ax2.set_yticklabels([])  # Remove y-ticks
    ax2.set_xticks(ticks=[0], labels=["Aggregate"])
    ax2.get_legend().remove()
    # Align y-axis limits between both plots
    ax2.set_ylim(ax1.get_ylim())

    # Improve layout
    plt.tight_layout()

    print(f"Saving to {save_path}")
    plt.savefig(save_path)
    plt.close()


def budget_plot(runs: list[dict]):
    """
    Plot mean classification accuracy for various budgets.
    Input arg looks like:
    [
        {
            "label": "llama3 70B", # Label for plotting. Also identifies the model
            "results": [
                (5, <dataset>),
                (10, <dataset>),
                ...
            ],
            "color": ... # Optional
        },
        {
            "label": "DeepSeek-r1 70B",
            "results": [...],
            "color": ...
        },
        ...
    ]
    """
    # Initialize seaborn
    sns.set_theme(style="darkgrid")
    sns.set_palette("Set2")
    sns.set_context("notebook")

    #### Keeping it bespoke for now
    run_group = "critic_200"

    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", 'exp-1', run_group)
    SAVE_DIR = os.path.join(RESULTS_DIR, "figs")
    save_fmt = "pdf"

    # Compile runs iterating over the RESULTS_DIR
    runs = []
    for run in os.listdir(RESULTS_DIR):
        if os.path.isdir(os.path.join(RESULTS_DIR, run)):
            run_config = json.load(open(os.path.join(RESULTS_DIR, run, "config.json"), 'r'))
            runs.append({
                "label": run_config["base_llm"]["name"],
                "budget": run_config["budget_per_task"],
                "results": json.load(open(os.path.join(RESULTS_DIR, run, "llm_results.json"), 'r'))
            })

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))

    for run in runs:
        label = run["label"]
        results = run["results"]
        color = run.get("color", None)

        # Extract the budgets and accuracies
        budgets = [r[0] for r in results]
        accuracies = [r[1]["accuracy"] for r in results]

        # Plot
        ax.plot(budgets, accuracies, label=label, color=color)

    # Add labels
    ax.set_xlabel("Budget")
    ax.set_ylabel("Mean Classification Accuracy")
    ax.legend(title=None)
    plt.tight_layout()

    # Save
    # Create figs directory if needed
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, "budget_plot.pdf")
    print(f"Saving to {save_path}")
    plt.savefig(save_path)
    plt.close()


if __name__ == '__main__':
    import sys
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    from sklearn.metrics import classification_report

    if len(sys.argv) < 3:
        raise ValueError(
            "Please provide a function name to call as first argument, and results name "
            "(relative to results/ directory) as second"
        )

    # Accept first argument as function name to be called
    func_name = sys.argv[1]
    if func_name not in globals():
        raise ValueError(f"Function {func_name} not found")

    # And second argument as the path to the results directory
    _results_name = sys.argv[2]

    print(f"Calling {func_name}")
    globals()[func_name](_results_name)
