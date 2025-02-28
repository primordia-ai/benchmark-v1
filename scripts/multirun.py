"""
Script for running multiple runs
"""
import subprocess
import time


def run_multi_exp1():
    runs = {
        "llama70b-budget15": [
            "--budget=15"
        ],
        "llama70b-budget30": [
            "--budget=30"
        ],
        "llama70b-budget60": [
            "--budget=60"
        ],
        "llama70b-budget120": [
            "--budget=120"
        ],
    }

    base_command = (
        "python experiment-1.py --no-input --test --llm-code=llama-large --extractor=llama-small --mode=both"
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
            break


if __name__ == '__main__':
    run_multi_exp1()
