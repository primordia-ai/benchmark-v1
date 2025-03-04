"""
Script for running multiple runs
"""
import subprocess
import time


def run_multi_exp1():
    extractor = "llama-small"
    tag = ""

    budget_iters = [
        10, 20
    ]
    model_iters = [
        # "llama-large",
        # "deepseek-r1-14b",
        "deepseek-r1-70b",
        # "qwen-32b",
        # "qwen-72b",
    ]

    runs = {}
    for model in model_iters:
        for budget in budget_iters:
            name = f"{tag}{model}-budget{budget}"
            runs[name] = [
                f"--llm-code={model}",
                f"--budget={budget}"
            ]

    base_command = (
        f"python experiment-1.py --no-input --extractor={extractor} --mode=llm"
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
