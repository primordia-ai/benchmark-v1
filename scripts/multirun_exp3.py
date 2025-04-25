"""
Script for running multiple runs
"""
import subprocess
import time
from cristal.env import PROJECT_ROOT


def run_multi_exp2():
    ############
    # Parameters
    actors = [
        #"llama-large",
        #"llama-small",
        #"mistral-small",
        #"mistral-nemo",
        #"deepseek-r1-14b",
        #"deepseek-r1-70b",
        #"qwen-32b",
        # "qwen-72b",
        # CAREFUL with rated usage
        "gpt-4o",
        #"gemini2",
        #"gemini25",
    ]

    critic = "gpt-4o"
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
    for actor in actors:
        for task in tasks:
            task_code = task["task"]
            difficulty = task["difficulty"]

            # Construct the name based on task and difficulty
            name = f"{task_code}/{difficulty}/{tag}{critic}-{actor}-{budget}"
            runs[name] = [
                f"--actor={actor}",
                f"--critic={critic}",
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
            print(f"Command failed with return code {result.returncode}")
            # Try the next one
            continue


if __name__ == '__main__':
    run_multi_exp2()
