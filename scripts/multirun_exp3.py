"""
Script for running multiple runs
"""
import subprocess
import time
from cristal.env import PROJECT_ROOT
from cristal.llm import get_llm


def run_multi_exp3():
    ############
    # Parameters
    actors = [
        "llama-large",
        "llama-small",
        "mistral-small",
        "mistral-nemo",
        "deepseek-r1:14b",  # small
        "deepseek-r1:70b", # medium
        "qwen2.5-coder:32b",
        "qwen2.5:72b",
        # CAREFUL with rated usage
        "gpt-4o",
        "gemini1.5",
        "gemini2",
        "gemini25",
    ]
    # Convert actor codes into paths codes
    path_codes = [
        get_llm(actor).path_code for actor in actors
    ]

    critic = "gpt-4o"
    critic_temp = 0.3
    actor_temp = 0.6

    tasks = [
        # Iterating over task code and difficulty combinations
        {
            "task": "inference_full",
            "difficulty": "hard"
        },
    ]
    max_loop_iters = 10
    budget = 600  # ten minutes
    seed = 101010

    # We'll override the reports dir to 200 for this run
    #reports_dir = str(PROJECT_ROOT) + "/data/reports/llama3_70b"

    tag = ""
    if tag:
        tag = tag.split("-")[0] + "-"
    runs = {}
    for actor, path_code in zip(actors, path_codes):
        for task in tasks:
            task_code = task["task"]
            difficulty = task["difficulty"]

            # Construct the name based on task and difficulty
            if tag:
                name = f"{task_code}/{difficulty}/{tag}/{critic}-{budget}/{path_code}"
            else:
                name = f"{task_code}/{difficulty}/{critic}-{budget}/{path_code}"
            runs[name] = [
                f"--actor={actor}",
                f"--actor-temp={actor_temp}",
                f"--critic={critic}",
                f"--critic-temp={critic_temp}",
                f"--task={task_code}",
                f"--difficulty={difficulty}",
            ]

    base_command = (
        f"python experiment-3.py --no-input --seed={seed} --max-iters={max_loop_iters} --budget={budget}"
    )

    for name in runs:
        command = base_command + f" --name={name}"
        for arg in runs[name]:
            command += f" {arg}"

        print(f"Running: {command}")
        time.sleep(5)
        result = subprocess.run(command, shell=True)
        if result.returncode != 0:
            print(f"Command failed with return code {result.returncode}. Continuing to next run.")
            # Try the next one
            continue


if __name__ == '__main__':
    run_multi_exp3()
