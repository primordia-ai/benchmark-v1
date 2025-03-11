"""
Script for running multiple runs
"""
import subprocess
import time
from cristal.env import PROJECT_ROOT


def run_multi_exp1():

    ############
    # Variables
    budget_iters = [
        20
    ]
    model_iters = [
        "llama-large",
        "llama-small",
        # "deepseek-r1-14b",
        # "deepseek-r1-70b",
        # "qwen-32b",
        # "qwen-72b",
        # CAREFUL with rated usage
        # "gpt-4o",
        # "gemini2",
    ]

    ############
    # Constants
    tag = "critic"
    if tag:
        tag = tag.split("-")[0] + "-"

    extractor = "llama-small"
    mode = "llm"
    # We'll override the reports dir to 200 for this run
    reports_dir = str(PROJECT_ROOT) + "/data/reports/llama3_70b_200"

    runs = {}
    for model in model_iters:
        for budget in budget_iters:
            name = f"{tag}{model}-budget{budget}"
            runs[name] = [
                f"--llm-code={model}",
                f"--budget={budget}",
            ]

    base_command = (
        f"python experiment-1.py --no-input --extractor={extractor} --mode={mode} --reports-dir={reports_dir}"
    )

    for name in runs:
        command = base_command + f" --name={name}"
        for arg in runs[name]:
            command += f" {arg}"

        print(f"Running: {command}")
        time.sleep(5)
        result = subprocess.run(command, shell=True)
        if result.returncode != 0:
            print(f"Command failed with return code {result.returncode}")
            # Try the next one
            continue


if __name__ == '__main__':
    run_multi_exp1()
