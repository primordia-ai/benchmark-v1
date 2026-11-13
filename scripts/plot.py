import json
import os

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import matthews_corrcoef, accuracy_score, classification_report, confusion_matrix
import numpy as np
from cristal.env import PROJECT_ROOT


def results_to_df(results_dir: str):
    """
    Return dataframe with CRISTAL, LLM, and Truth results within
    """
    llm_results = cristal_results = truth_results = {}
    cristal_path: str = os.path.join(results_dir, "cristal_results.json")
    if os.path.exists(cristal_path):
        cristal_results: dict = json.load(open(cristal_path, 'r'))

    llm_path: str = os.path.join(results_dir, "llm_results.json")
    # Load the json if it exists
    if os.path.exists(llm_path):
        llm_results: dict = json.load(open(llm_path, 'r'))

    truth_path: str = os.path.join(results_dir, "truth_data.json")
    if os.path.exists(truth_path):
        truth_results: dict = json.load(open(truth_path, 'r'))

    # Combine the set of errored assets from both CRISTAL and LLM, to keep things balanced
    errors = set()
    errors |= {a_id for a_id, asset in cristal_results.items() if "error" in asset}
    errors |= {a_id for a_id, asset in llm_results.items() if "error" in asset}

    # Process CRISTAL results -- filter out errors
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
            "category": asset["category"],
            "growth": asset["growth"],
        }
        for a_id, asset in truth_results.items() if a_id not in errors
    ]

    # Create a combined pandas df of the category and growth values for llm and truth sets
    datasets = []
    if cristal_data:
        datasets.append(pd.DataFrame(cristal_data).assign(source="CRISTAL"))
    if llm_data:
        datasets.append(pd.DataFrame(llm_data).assign(source="LLM"))
    if truth_data:
        datasets.append(pd.DataFrame(truth_data).assign(source="Truth"))

    if datasets:
        df = pd.concat(datasets, ignore_index=True)
        return df
    else:
        raise ValueError("No valid data to concatenate. Ensure the input files are not empty or missing.")


def exp1(results_name: str):
    # ================
    # Some setup

    # Initialize seaborn
    sns.set_theme(style="darkgrid")
    sns.set_palette("colorblind")
    sns.set_context("paper")

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
    df = results_to_df(RESULTS_DIR)

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

def exp2(results_name: str):
    # ================
    # Some setup
    print("Running exp2")
    # Initialize seaborn
    sns.set_theme(style="darkgrid")
    sns.set_palette("colorblind")
    sns.set_context("paper")

    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", 'exp-2', results_name)
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
        cristal_results = json.load(open(os.path.join(RESULTS_DIR, "results.json"), 'r'))

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


def exp3(results_name: str):
    """
    Plot the resources used for the actor-critic performance of different models

    Since the metric is binary, should probably just do a table with columns:
        - Iters used
        - Average budget per iter
        - Approved boolean

    Args:
        results_name: A directory containing containing sub-directories of actor models.
            Each actor subdir has a 'results.json' file, is a list of data from each actor-critic loop
            [
                [
                    {
                        "approved": <bool>,
                        "actor_delta": <float>,  # Actor time delta for that iter
                        "eval_delta": <float>,  # Critic delta
                        "iter_delta": <float>,  # Combined time of actor + critic
                    }
                ],
                [
                    {...},
                ],
                ...
            ]
    """
    # Initialize seaborn
    sns.set_theme(style="darkgrid")
    sns.set_palette("colorblind")
    sns.set_context("paper")

    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", 'exp-3', results_name)
    SAVE_DIR = os.path.join(RESULTS_DIR, "figs")
    save_fmt = "pdf"

    # Create figs directory
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load results from each actor
    all_results = {}
    for actor_dir in os.listdir(RESULTS_DIR):
        if actor_dir.startswith("."):
            continue  # Skip hidden files
        actor = str(actor_dir)
        actor_path = os.path.join(RESULTS_DIR, actor_dir)
        print(f"Loading {actor_path}")
        if not os.path.isdir(actor_path):
            continue  # Skip non-directories
        try:
            with open(os.path.join(actor_path, "results.json"), 'r') as f:
                data = json.load(f)
                # Remove unnecessary inner bracketting
                data = [d[0] for d in data]
                final = data[-1]
                all_results[actor] = {
                    "approved": final['approved'],
                    "actor_delta": int(np.mean([d['actor_delta'] for d in data])),
                    "critic_delta": int(np.mean([d['eval_delta'] for d in data])),
                    "iter_delta": int(np.mean([d['iter_delta'] for d in data])),
                    "num_iters": len(data)
                }
        except FileNotFoundError:
            print(f"File not found: {actor_path}/results.json")
            continue

    # Combine the results into a single DataFrame
    rows = []
    for actor, result in all_results.items():
        rows.append({
            "model": actor,
            "approved": result["approved"],
            "actor_delta": result["actor_delta"],
            "critic_delta": result["critic_delta"],
            "iter_delta": result["iter_delta"],
            "num_iters": result["num_iters"],
        })

    # Dataframe
    df = pd.DataFrame(rows)
    # Sort by: Approved==True, Num Iters, Actor Delta
    df = df.sort_values(by=["approved", "num_iters", "actor_delta"], ascending=[False, True, True])

    # Plot the results in tabular format
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('tight')
    ax.axis('off')

    # Create a table from the DataFrame
    col_mapping = {
        "model": "Model",
        "approved": "Approved",
        "actor_delta": "Actor Delta (s)",
        "critic_delta": "Critic Delta (s)",
        "iter_delta": "Iter Delta (s)",
        "num_iters": "Num Iters",
    }
    table = ax.table(cellText=df.values, colLabels=[col_mapping[c] for c in df.columns], cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    # Improve layout
    plt.tight_layout()

    # Latex Table
    latex_table_path = os.path.join(SAVE_DIR, "table1.tex")
    cols = ["Model", "Approved", "Num Iters", "Actor Delta"]

    with open(latex_table_path, "w") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{|l|l|r|r|}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Model} & \\textbf{Approved} & \\textbf{Num Iters} & \\textbf{Actor Delta (s)} \\\\ \n")
        f.write("\\noalign{\\hrule height 2pt}\n")

        # n_shots_counts = results_df['n-shots'].value_counts().to_dict()  # Count occurrences of each n-shot value
        for _, row in df.iterrows():
            f.write(f"{row['model']} & {row['approved']} & {row['num_iters']} & {row['actor_delta']} \\\\ \n")
            # if row['n-shots'] != current_n_shots:
            #     current_n_shots = row['n-shots']
            #     row_span = n_shots_counts[current_n_shots]  # Get the number of rows for this n-shot value
            #     if current_n_shots != list(n_shots_counts.keys())[0]:
            #         f.write("\\noalign{\\hrule height 2pt}\n")  # Use noalign with hrule for a thicker horizontal line for new multirow, except for the first one
            #     f.write(f"\\multirow{{{row_span}}}{{*}}{{{current_n_shots}}} & {row['Model']} & {row['Accuracy']} & {row['MCC']} \\\\ \\cline{{2-4}}\n")
            # else:
            #     f.write(f" & {row['Model']} & {row['Accuracy']} & {row['MCC']} \\\\ \\cline{{2-4}}\n")

        f.write("\\hline\n")  # Add final \hline at the end of the table
        f.write("\\end{tabular}\n")
        f.write("\\caption{Results Table of }\n")
        f.write("\\label{tab:results_table}\n")
        f.write("\\end{table}\n")

    save_path = os.path.join(SAVE_DIR, f"fig1.{save_fmt}")
    print(f"Saving to {save_path}")
    plt.savefig(save_path)
    plt.close()


def budget_plot():
    """
    Plot mean classification accuracy for various budgets.
    """
    # Initialize seaborn
    sns.set_theme(style="darkgrid")
    sns.set_palette("colorblind")
    sns.set_context("paper")

    #### Keeping it bespoke for now
    run_group = "critic_200"
    cristal_runs = [
        "cristal-budget5",
        "cristal-budget10",
        "cristal-budget20",
        "cristal-budget60",
    ]
    llm_runs = [
        "deepseek-r1-70b-budget5",
        # "deepseek-r1-70b-budget10",
        "deepseek-r1-70b-budget20",
        "deepseek-r1-70b-budget60",
        # "deepseek-r1-70b-budget120",
        "llama-large-budget5",
        "llama-large-budget10",
        "llama-large-budget20",
        "llama-large-budget60",
        "2-mistral-small-budget5",
        "2-mistral-small-budget10",
        "2-mistral-small-budget20",
        "2-mistral-small-budget60",
        "2-mistral-nemo-budget5",
        "2-mistral-nemo-budget10",
        "2-mistral-nemo-budget20",
        "2-mistral-nemo-budget60",
    ]

    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", 'exp-1', run_group)
    SAVE_DIR = os.path.join(RESULTS_DIR, "figs")
    save_fmt = "pdf"

    # Compile runs iterating over the RESULTS_DIR
    results = {}
    for c_run in cristal_runs:
        run_path = os.path.join(RESULTS_DIR, c_run)
        run_config = json.load(open(os.path.join(run_path, "config.json"), 'r'))
        run_name = "CRISTAL"
        budget = run_config["budget_per_task"]
        # Load data
        df = results_to_df(run_path)
        y_true = df[df["source"] == "Truth"].sort_values("asset")["category"]
        y_pred_cristal = df[df["source"] == "CRISTAL"].sort_values("asset")["category"]
        # Calculate accuracy and MCC
        accuracy_cristal = (y_true.reset_index(drop=True) == y_pred_cristal.reset_index(drop=True)).mean()
        mcc_cristal = matthews_corrcoef(y_true, y_pred_cristal)
        # Append to data of this run name or create new
        if run_name not in results:
            results[run_name] = {
                "accuracy": {
                    "x": [float(budget)],
                    "y": [accuracy_cristal],
                },
                "mcc": {
                    "x": [float(budget)],
                    "y": [mcc_cristal],
                }
            }
        else:
            results[run_name]['accuracy']['x'].append(budget)
            results[run_name]['accuracy']['y'].append(accuracy_cristal)
            results[run_name]['mcc']['x'].append(budget)
            results[run_name]['mcc']['y'].append(mcc_cristal)

    for l_run in llm_runs:
        run_path = os.path.join(RESULTS_DIR, l_run)
        run_config = json.load(open(os.path.join(run_path, "config.json"), 'r'))
        run_name = run_config["base_llm"]["name"]
        budget = run_config["budget_per_task"]
        # Load data
        df = results_to_df(run_path)
        y_true = df[df["source"] == "Truth"].sort_values("asset")["category"]
        y_pred_llm = df[df["source"] == "LLM"].sort_values("asset")["category"]
        # Calculate accuracy and MCC
        accuracy_llm = (y_true.reset_index(drop=True) == y_pred_llm.reset_index(drop=True)).mean()
        mcc_llm = matthews_corrcoef(y_true, y_pred_llm)
        # Append to data of this run name or create new
        if run_name not in results:
            results[run_name] = {
                "accuracy": {
                    "x": [float(budget)],
                    "y": [accuracy_llm],
                },
                "mcc": {
                    "x": [float(budget)],
                    "y": [mcc_llm],
                }
            }
        else:
            results[run_name]['accuracy']['x'].append(budget)
            results[run_name]['accuracy']['y'].append(accuracy_llm)
            results[run_name]['mcc']['x'].append(budget)
            results[run_name]['mcc']['y'].append(mcc_llm)

    # Create two plots for the Accuracy and MCC
    fig, axs = plt.subplots(2, 1, figsize=(6, 8))
    fig.suptitle("N-shot Analysis", fontsize=16)  # Add a title to the figure

    from pprint import pprint
    pprint(results)

    # Accuracy plot
    ax = axs[0]
    for name, data in results.items():
        ax.plot(data['accuracy']['x'], data['accuracy']['y'], label=name, marker='x', linestyle=None)
    # X-axis
    ax.set_xticklabels([])
    ax.set_xlabel('')
    # Y-axis
    ax.set_ylabel("Mean Classification Accuracy")
    ax.set_ylim(0, 1)
    ax.legend(title=None)

    # MCC plot
    ax = axs[1]
    for name, data in results.items():
        ax.plot(data['mcc']['x'], data['mcc']['y'], label=name, marker='x', linestyle=None)
    # X-axis
    ax.set_xlabel("Budget")
    # Y-axis
    ax.set_ylabel("Mean MCC")
    # ax.set_ylim(-1, 1)
    ax.legend(title=None)

    # Tight layout
    plt.tight_layout()

    # Save
    # Create figs directory if needed
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, "budget_plot.pdf")
    print(f"Saving to {save_path}")
    plt.savefig(save_path)
    plt.close()


def n_shot_plot(results_name: str):
    """
    Plot mean classification accuracy for various n-shot values.
    """

    # Initialize seaborn
    sns.set_theme(style="darkgrid")
    sns.set_palette("colorblind")
    sns.set_context("paper")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", results_name)
    SAVE_DIR = os.path.join(RESULTS_DIR, "figs")
    save_fmt = "pdf"

    # Compile runs iterating over the RESULTS_DIR
    results = {}
    for run_dir in os.listdir(RESULTS_DIR):
        run_path = os.path.join(RESULTS_DIR, run_dir)
        run_config = None
        if not os.path.isdir(run_path):
            continue
        # Infer n_shot from the directory name if not present in config.json
        try:
            run_config = json.load(open(os.path.join(run_path, "config.json"), 'r'))
        except FileNotFoundError:
            print(f"Skipping {run_path}, no config.json found.")
            continue
        n_shot = run_config.get("n_shot", None)
        if n_shot is None:
            # Attempt to extract n_shot from the directory name
            try:
                n_shot = int(run_dir.split("n_shots")[1])
            except (ValueError, IndexError):
                print(f"Skipping {run_path}, unable to infer n_shot value.")
                continue

        llm_code = run_config["base_llm"]["code"]
        # Load data
        print(run_path)
        df = results_to_df(run_path)
        # For LLM
        y_true_llm = df[df["source"] == "Truth"].sort_values("asset")[["category"]]
        y_pred_llm = df[df["source"] == "LLM"].sort_values("asset")[["category"]]
        # Calculate accuracy and MCC for LLM
        y_true_llm = y_true_llm.reset_index(drop=True)
        y_pred_llm = y_pred_llm.reset_index(drop=True)

        try:
            accuracy_llm = accuracy_score(y_true_llm, y_pred_llm)
            mcc_llm = matthews_corrcoef(y_true_llm.values.ravel(), y_pred_llm.values.ravel())
        except ValueError as e:
            print(f"Skipping {run_path}, unable to calculate accuracy or MCC for LLM: {e}")
            continue

        # For CRISTAL
        if "CRISTAL" in df["source"].unique():
            y_true_cristal = df[df["source"] == "Truth"].sort_values("asset")[["category"]]
            y_pred_cristal = df[df["source"] == "CRISTAL"].sort_values("asset")[["category"]]
            # Calculate accuracy and MCC for CRISTAL
            y_true_cristal = y_true_cristal.reset_index(drop=True)
            y_pred_cristal = y_pred_cristal.reset_index(drop=True)
            accuracy_cristal = accuracy_score(y_true_cristal, y_pred_cristal)
            mcc_cristal = matthews_corrcoef(y_true_cristal.values.ravel(), y_pred_cristal.values.ravel())
        else:
            accuracy_cristal = mcc_cristal = np.nan

        # else:
        #     y_true_cristal = pd.DataFrame(columns=["category"])
        #     y_pred_cristal = pd.DataFrame(columns=["category"])
        #
        # try:
        #     accuracy_cristal = accuracy_score(y_true_cristal, y_pred_cristal)
        #     mcc_cristal = matthews_corrcoef(y_true_cristal.values.ravel(), y_pred_cristal.values.ravel())
        # except ValueError as e:
        #     accuracy_cristal = np.nan
        #     mcc_cristal = np.nan
        #     print(f" {run_path}, unable to calculate accuracy or MCC for CRISTAL: {e}, assigning NaN")

        # Append to data of this run name or create new
        if llm_code not in results:
            results[llm_code] = {
            "accuracy": {
                "x": [float(n_shot)],
                "llm_y": [accuracy_llm],
                "cristal_y": [accuracy_cristal],
            },
            "mcc": {
                "x": [float(n_shot)],
                "llm_y": [mcc_llm],
                "cristal_y": [mcc_cristal],
            }
            }
        else:
            results[llm_code]['accuracy']['x'].append(n_shot)
            results[llm_code]['accuracy']['llm_y'].append(accuracy_llm)
            results[llm_code]['accuracy']['cristal_y'].append(accuracy_cristal)
            results[llm_code]['mcc']['x'].append(n_shot)
            results[llm_code]['mcc']['llm_y'].append(mcc_llm)
            results[llm_code]['mcc']['cristal_y'].append(mcc_cristal)

    # Sorted models. Code to styles.
    # Grab 10 colors
    colors = sns.color_palette("colorblind", 10)
    ordered_models = [
        (
            "cristal",
            {
                "label": "CRISTAL",
                "marker": "d",
                "color": colors[0],
                "ls": "-",
            }
        ),
        (
            "deepseek-r1:14b",
            {
                "label": "Deepseek-R1 14B",
                "marker": "^",
                "color": colors[1],
                "ls": ":",
            }
        ),
        (
            "deepseek-r1:70b",
            {
                "label": "Deepseek-R1 70B",
                "marker": "^",
                "color": colors[1],
                "ls": "-",
            }
        ),
        (
            "gpt-4o-2024-08-06",
            {
                "label": "GPT-4o",
                "marker": "o",
                "color": colors[2],
                "ls": "-",
            }
        ),
        (
            "llama3.1:8b",
            {
                "label": "Llama 3.1 8B",
                "marker": "s",
                "color": colors[3],
                "ls": ":",
            }
        ),
        (
            "llama3:70b",
            {
                "label": "Llama 3 70B",
                "marker": "s",
                "color": colors[3],
                "ls": "-",
            }
        ),
        (
            "mistral-nemo",
            {
                "label": "Mistral Nemo 3",
                "marker": "p",
                "color": colors[4],
                "ls": ":",
            }
        ),
        (
            "mistral-small",
            {
                "label": "Mistral Small 3",
                "marker": "p",
                "color": colors[4],
                "ls": "-",
            }
        ),
        (
            "qwen2.5-coder:32b",
            {
                "label": "Qwen2.5-Coder 32B",
                "marker": "h",
                "color": colors[6],
                "ls": ":",
            }
        ),
        (
            "qwen2.5:72b",
            {
                "label": "Qwen2 72B",
                "marker": "h",
                "color": colors[6],
                "ls": "-",
            }
        ),
    ]

    fig, axs = plt.subplots(1, 2, figsize=(9, 3))

    #####  ACCURACY PLOT

    ax = axs[0]
    # CRISTAL Plot
    cristal_data = {
        "accuracy": {
            "x": [],
            "cristal_y": []
        },
        "mcc": {
            "x": [],
            "cristal_y": []
        }
    }
    for name, data in results.items():
        cristal_data["accuracy"]["x"].extend(data["accuracy"]["x"])
        cristal_data["accuracy"]["cristal_y"].extend(data["accuracy"]["cristal_y"])
        cristal_data["mcc"]["x"].extend(data["mcc"]["x"])
        cristal_data["mcc"]["cristal_y"].extend(data["mcc"]["cristal_y"])

    sorted_cristal_data = sorted(zip(cristal_data['accuracy']['x'], cristal_data['accuracy']['cristal_y']))
    x_sorted_cristal = [x for x, _ in sorted_cristal_data]
    cristal_means = [pd.Series([y for x_val, y in sorted_cristal_data if x_val == x]).mean() for x in sorted(set(x_sorted_cristal))]
    cristal_stds = [pd.Series([y for x_val, y in sorted_cristal_data if x_val == x]).std() for x in sorted(set(x_sorted_cristal))]
    x_unique = sorted(set(x_sorted_cristal))

    crs_styles = ordered_models[0][1]
    sns.lineplot(
        x=x_unique,
        y=cristal_means,
        label="CRISTAL",
        marker=crs_styles["marker"],
        linestyle=crs_styles["ls"],
        color=crs_styles["color"],
        ax=ax,
        legend=False,
    )

    # LLM Plots

    for code, styles in ordered_models:
        if code not in results:
            continue

        data = results[code]
        # Sort the results by x values (n_shot) for consistent plotting
        sorted_data = sorted(zip(data['accuracy']['x'], data['accuracy']['llm_y']))
        x_sorted, llm_y_sorted = zip(*sorted_data)
        # color = plt.cm.Set1(len(ax.lines) % 10)  # Use the number of existing lines to assign unique colors
        sns.lineplot(
            x=x_sorted,
            y=llm_y_sorted,
            label=styles["label"],
            marker=styles["marker"],
            linestyle=styles["ls"],
            color=styles["color"],
            ax=ax,
            legend=False,
        )
        # ax.plot(x_sorted, llm_y_sorted, label=label, marker=marker, linestyle='--', color=color)


    # Plot Styling
    ax.set_xticks(sorted(set(x for data in results.values() for x in data['accuracy']['x'])))
    ax.set_xlabel("N-shot examples per category")
    ax.set_ylabel("Classification Accuracy")
    ax.set_ylim(0, 1)

    ######  MCCC PLOT

    ax = axs[1]
    # CRISTAL Plot
    sorted_cristal_data = sorted(zip(cristal_data['mcc']['x'], cristal_data['mcc']['cristal_y']))
    x_sorted_cristal = [x for x, _ in sorted_cristal_data]
    cristal_means = [pd.Series([y for x_val, y in sorted_cristal_data if x_val == x]).mean() for x in sorted(set(x_sorted_cristal))]
    cristal_stds = [pd.Series([y for x_val, y in sorted_cristal_data if x_val == x]).std() for x in sorted(set(x_sorted_cristal))]
    x_unique = sorted(set(x_sorted_cristal))
    sns.lineplot(
        x=x_unique,
        y=cristal_means,
        label="CRISTAL",
        marker=crs_styles["marker"],
        linestyle=crs_styles["ls"],
        color=crs_styles["color"],
        ax=ax,
        legend=False,
    )

    # LLM Plots

    for code, style in ordered_models:
        if code not in results:
            continue

        data = results[code]
        # Sort the results by x values (n_shot) for consistent plotting
        sorted_data = sorted(zip(data['mcc']['x'], data['mcc']['llm_y']))
        x_sorted, llm_y_sorted = zip(*sorted_data)
        # color = plt.cm.Set1(len(ax.lines) % 10)  # Use the number of existing lines to assign unique colors
        # ax.plot(x_sorted, llm_y_sorted, label=f"{name}", marker='x', linestyle='--', color=color)
        sns.lineplot(
            x=x_sorted,
            y=llm_y_sorted,
            label=style["label"],
            marker=style["marker"],
            linestyle=style["ls"],
            color=style["color"],
            ax=ax,
            legend=False,
        )

    # Some styling
    ax.set_xticks(sorted(set(x for data in results.values() for x in data['mcc']['x'])))
    ax.set_xlabel("N-shot examples per category")
    ax.set_ylabel("MCC")

    #### Final Figure styling

    # Add legends outside the plots
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", title=None, bbox_to_anchor=(1, 0.5))

    # Tight layout
    plt.tight_layout(rect=[0, 0, 0.8, 1])  # Adjust layout to make space for the legend
    # plt.tight_layout()  # Adjust layout to make space for the legend
    # Save
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, "n_shot_plot.pdf")
    print(f"Saving to {save_path}")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    ###### LaTeX TABLE

    # Convert results to a pandas DataFrame and generate a LaTeX table
    results_df = pd.DataFrame()

    # Process LLM results
    cristal_data = {
        "n-shots": [],
        "Accuracy": [],
        "MCC": []
    }
    for code, style in ordered_models:
        if code not in results:
            continue

        data = results[code]
        for i, x in enumerate(data["accuracy"]["x"]):
            results_df = pd.concat([
                results_df,
                pd.DataFrame({
                    "Model": [style["label"]],
                    "n-shots": [int(x)],
                    "Accuracy": [data["accuracy"]["llm_y"][i]],
                    "MCC": [data["mcc"]["llm_y"][i]]
                })
            ], ignore_index=True)

        cristal_data["n-shots"].extend(data["accuracy"]["x"])
        cristal_data["Accuracy"].extend(data["accuracy"]["cristal_y"])
        cristal_data["MCC"].extend(data["mcc"]["cristal_y"])

    unique_n_shots = sorted(set(cristal_data["n-shots"]))
    for n_shot in unique_n_shots:
        accuracy_mean = pd.Series(
            [acc for n, acc in zip(cristal_data["n-shots"], cristal_data["Accuracy"]) if n == n_shot]
        ).mean()
        mcc_mean = pd.Series(
            [mcc for n, mcc in zip(cristal_data["n-shots"], cristal_data["MCC"]) if n == n_shot]
        ).mean()
        results_df = pd.concat([
            results_df,
            pd.DataFrame({
                "Model": ["CRISTAL"],
                "n-shots": [int(n_shot)],
                "Accuracy": [accuracy_mean],
                "MCC": [mcc_mean]
            })
        ], ignore_index=True)

    # Save the DataFrame as a LaTeX table
    latex_table_path = os.path.join(SAVE_DIR, "results_table.tex")
    print(f"Saving LaTeX table to {latex_table_path}")
    results_df = results_df.sort_values(by=["n-shots", "Model"])
    # Highlight the highest values per numerical column with bold font
    for col in ["Accuracy", "MCC"]:
        if col in results_df.columns:
            max_values = results_df.groupby(["n-shots"])[col].transform("max")
            results_df[col] = results_df.apply(
                lambda row: f"\\textbf{{{row[col]:.2f}}}" if row[col] == max_values[row.name] else f"{row[col]:.2f}",
                axis=1
            )
    # Reorder columns to place "n-shots" as the first column
    results_df = results_df.sort_values(by=["n-shots", "Model"])
    # Reorder columns to place "n-shots" as the first column
    results_df = results_df[["n-shots", "Model", "Accuracy", "MCC"]]

    # Create a multirow design for n-shots
    results_df["n-shots"] = results_df["n-shots"].astype(str)  # Ensure n-shots is treated as a string for grouping

    with open(latex_table_path, "w") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{|l|l|r|r|}\n")
        f.write("\\hline\n")
        f.write("\\textbf{N-shots} & \\textbf{Model} & \\textbf{Accuracy} & \\textbf{MCC} \\\\ \n")
        f.write("\\noalign{\\hrule height 2pt}\n")
        current_n_shots = None
        n_shots_counts = results_df['n-shots'].value_counts().to_dict()  # Count occurrences of each n-shot value
        for _, row in results_df.iterrows():
            if row['n-shots'] != current_n_shots:
                current_n_shots = row['n-shots']
                row_span = n_shots_counts[current_n_shots]  # Get the number of rows for this n-shot value
                if current_n_shots != list(n_shots_counts.keys())[0]:
                    f.write("\\noalign{\\hrule height 2pt}\n")  # Use noalign with hrule for a thicker horizontal line for new multirow, except for the first one
                f.write(f"\\multirow{{{row_span}}}{{*}}{{{current_n_shots}}} & {row['Model']} & {row['Accuracy']} & {row['MCC']} \\\\ \\cline{{2-4}}\n")
            else:
                f.write(f" & {row['Model']} & {row['Accuracy']} & {row['MCC']} \\\\ \\cline{{2-4}}\n")

        f.write("\\hline\n")  # Add final \hline at the end of the table
        f.write("\\end{tabular}\n")
        f.write("\\caption{Results Table for N-shot Analysis}\n")
        f.write("\\label{tab:results_table}\n")
        f.write("\\end{table}\n")


if __name__ == '__main__':
    import sys
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    from sklearn.metrics import classification_report

    # if len(sys.argv) < 3:
    #     raise ValueError(
    #         "Please provide a function name to call as first argument, and results name "
    #         "(relative to results/ directory) as second"
    #     )

    # Accept first argument as function name to be called
    func_name = sys.argv[1]
    if func_name not in globals():
        raise ValueError(f"Function {func_name} not found")
    print(f"Calling {func_name}")

    if func_name == "exp1":
        # And second argument as the path to the results directory
        _results_name = sys.argv[2]
        globals()[func_name](_results_name)

    if func_name == "exp2":
        # And second argument as the path to the results directory
        _results_name = sys.argv[2]
        globals()[func_name](_results_name)

    elif func_name == "budget_plot":
        globals()[func_name]()

    elif func_name == "n_shot_plot":
        _results_name = sys.argv[2]
        globals()[func_name](_results_name)

    elif func_name == "exp3":
        _results_name = sys.argv[2]
        globals()[func_name](_results_name)

    # elif func_name == "exp4_n_shot_plot":
    #     globals()[func_name]()