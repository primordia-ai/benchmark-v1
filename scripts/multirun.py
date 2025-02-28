"""
Script for running multiple runs
"""
import subprocess
import time


def run_multi_exp1():
    llm_base = "llama-small"
    extractor = "llama-small"
    tag = "rankV2-"

    runs = {
        f"{tag}{llm_base}-budget5": [
            "--budget=5"
        ],
        f"{tag}{llm_base}-budget10": [
            "--budget=10"
        ],
        f"{tag}{llm_base}-budget15": [
            "--budget=15"
        ],
        f"{tag}{llm_base}-budget20": [
            "--budget=20"
        ],
        # f"{tag}{llm_base}-budget120": [
        #     "--budget=120"
        # ],
    }

    base_command = (
        f"python experiment-1.py --no-input --llm-code={llm_base} --extractor={extractor} --mode=both"
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
