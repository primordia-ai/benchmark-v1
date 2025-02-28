"""
Script for running multiple runs
"""
import subprocess
import time


def run_multi_exp1():
    llm_base = "llama-small"
    extractor = "llama-small"

    runs = {
        f"{llm_base}-budget15": [
            "--budget=15"
        ],
        f"{llm_base}-budget30": [
            "--budget=30"
        ],
        f"{llm_base}-budget60": [
            "--budget=60"
        ],
        f"{llm_base}-budget120": [
            "--budget=120"
        ],
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
