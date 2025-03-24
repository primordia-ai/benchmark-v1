"""
Script for running multiple runs
"""
import subprocess
import time
from cristal.env import PROJECT_ROOT


def run_multi_exp2():

    ############
    # Variables
    n_shot_iters = [
        1, 5, 10, 20, 60, 80
    ]
    model_iters = [
        #"llama-large",
        #"llama-small",
        #"mistral-small",
        #"mistral-nemo",
        "deepseek-r1-14b",
        "deepseek-r1-70b",
        "qwen-32b",
        "qwen-72b",
        # CAREFUL with rated usage
        # "gpt-4o",
        # "gemini2",
        #####
        # Special handling for cristal
        # "cristal"
    ]

    ############
    # Constants
    tag = "multirun/"
    # if tag:
    #     tag = tag.split("-")[0] + "-"

    extractor = "llama-small"
    mode = "both"
    # We'll override the reports dir to 200 for this run
    #reports_dir = str(PROJECT_ROOT) + "/data/reports/llama3_70b"

    runs = {}
    for model in model_iters:
        for n_shots in n_shot_iters:
            name = f"{tag}{model}-n_shots{n_shots}"
            if model == "cristal":
                runs[name] = [
                    f"--n-shots={n_shots}",
                ]
            else:
                runs[name] = [
                    f"--llm-code={model}",
                    f"--n-shots={n_shots}",
                ]

    base_command = (
        f"python experiment-2.py --no-input --mode={mode}"
    )
    if mode != "llm":
        base_command += f" --extractor={extractor}"

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
    run_multi_exp2()
