"""
Experiment 3

Through iterations on an actor-critic loop, can powerful LLMs arrive at correct program
solutions to our inference problem? And if so, how long does this take?

Metrics of interest are Number of iterations, Wall time (budget), Success, ...

Other data to track: The solution attempts, and the critic evaluation responses.
"""

import argparse
import json
import os
import sys
import shutil

import numpy as np
import pytensor
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from cristal import stats
from cristal.env import PROJECT_ROOT
from cristal.llm import get_llm, utils as llm_utils
from cristal.llm.codegen.model import LLMCodeGen
from cristal.llm.codegen.critic import CodeGenCritic
from cristal.llm.codegen import likelihood_inference_full
from cristal.logger import capture_description, log_to_file, logger, reset_logger


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3"
    )
    parser.add_argument(
        "--test", action="store_true", help="Run in test mode"
    )
    parser.add_argument(
        "--actor", type=str, help="Actor model that is also used for Codegen generation"
    )
    parser.add_argument(
        "--critic", type=str, help="Codegen Critic model"
    )
    parser.add_argument(
        "--task", type=str, help="Which Codegen task to run."
    )
    parser.add_argument(
        "--difficulty", type=str, default="hard", help="Task difficulty, ['easy', 'med', 'hard']. Default 'hard'."
    )
    parser.add_argument(
        "--name", type=str, default="latest", help="Experiment name. Default 'latest'."
    )
    parser.add_argument(
        "--seed", type=int, default=SEED, help=f"Random seed. Default {SEED}."
    )
    parser.add_argument(
        "--strict", action="store_true", help="Do not ignore exceptions on asset evaluation. By default this "
                                              "is excluded, so that other assets can continue to be evaluated."
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO", help="Logging level. Default 'INFO'."
    )
    parser.add_argument(
        "--no-input", action="store_true", help="Skip any input prompting"
    )
    parser.add_argument(
        "--budget", type=int, default=120, help="Budget per task in seconds. Default is two minutes."
    )
    parser.add_argument(
        "--max-iters", type=int, default=10, help="Maximum number of critic evaluation iters"
    )
    parser.add_argument(
        "--compress-reports", action="store_true", help="Attempt to compress the reports through recursive summarizing"
    )
    parser.add_argument(
        "--reports-dir", type=str, default=None, help="Reports directory override over env REPORTS_DIR."
    )

    args = parser.parse_args()

    tasks = {
        "inference_full": {
            "validator": likelihood_inference_full.LikelihoodInferenceFull,
            "easy": likelihood_inference_full.TASK_EASY,
            "hard": likelihood_inference_full.TASK_HARD,
        }
    }
    task_code = args.task
    task_diff = args.difficulty
    validator, task = tasks[task_code]["validator"], tasks[task_code][task_diff]

    # Seed rng
    seed = args.seed
    np.random.seed(seed)
    # Log level
    loglevel = args.log_level
    file_loglevel = "DEBUG" if loglevel == "DEBUG" else "INFO"
    if loglevel != "INFO":
        reset_logger(loglevel)

    # Run dir
    run_name = args.name
    results_dir = os.path.join(PROJECT_ROOT, f"results/exp-3/{run_name}")
    # Recursively remove dir if it exists
    if os.path.exists(results_dir):
        logger.info(f"Removing existing results directory {results_dir}")
        shutil.rmtree(results_dir)

    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "out.log")
    # Set logging to file
    log_to_file(log_path, level=file_loglevel)

    # Get some info about the run for adding to the config file
    no_input = args.no_input
    description = ""
    command_str = "python " + " ".join(sys.argv)
    if not no_input:
        # Prompt user for description
        desc_path = os.path.join(results_dir, "description.txt")
        description = capture_description(logger)

    testing = args.test
    strict = args.strict
    if testing:
        # Set retries and budget
        actor_retries, actor_budget = 3, 20
        loop_max_iters = 4
        loop_budget = args.budget
        # Set logger to debug

    else:
        # Set retries and budget
        actor_retries, actor_budget = 3, 60
        loop_max_iters = 10
        loop_budget = args.budget

    #####################
    # Configure models
    actor_code = args.actor
    critic_code = args.critic

    # Critic Model
    critic_log_path = os.path.join(results_dir, "critic.log")
    critic_llm = get_llm(critic_code)(seed=seed)
    # Activate logging LLM
    logger.info(f"Piping Critic logs to {critic_log_path}")
    critic_llm.log_to_file(critic_log_path)

    # Actor Model
    actor_log_path = os.path.join(results_dir, "actor.log")
    actor_llm = get_llm(actor_code)(seed=seed)
    # Activate logging LLM
    logger.info(f"Piping Actor logs to {actor_log_path}")
    actor_llm.log_to_file(actor_log_path)

    critic = CodeGenCritic(critic_llm, task)
    actor = LLMCodeGen(actor_llm, critic)

    #####################
    # Save details to config file
    run_data = {
        "command": command_str,
        "description": description,
        "seed": seed,
        "test": testing,
        "actor_llm": actor_llm.to_json(),
        "critic_llm": critic_llm.to_json(),
        "loop_budget": loop_budget,
        "loop_max_iters": loop_max_iters,
        "validator": validator.function_name,
        "validator_description": validator.prompt_description,
        "task_difficulty": task_diff,
        "task": task,
    }

    config_path = os.path.join(results_dir, "config.json")
    with open(config_path, 'w') as f:
        logger.info(f"Saving run data to {config_path}")
        json.dump(run_data, f, indent=4)

    # Call the loop method
    critic_result = actor.critic_loop(
        validator, loop_max_iters, loop_budget, actor_retries, actor_budget
    )
    # Print some details of the result
    logger.info("\nCritic loop finished:")
    logger.info(f"\tApproved: {critic_result['approved']}")
    logger.info(f"\tIters: {critic_result['iters']}")
    logger.info(f"\tCost: {float(critic_result['cost']):2f}")

    all_results = critic_result['all_results']
    logger.debug(f"Last result:\n\t{all_results[-1]}")

    # Save the bulk results
    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, 'w') as f:
        logger.info(f"Saving results to {results_path}")
        json.dump(all_results, f, indent=4)


if __name__ == '__main__':

    pytensor.config.exception_verbosity = 'high'

    # Set seed
    # 101010 happens to be 42 in binary. How charming.
    SEED = 101010

    main()
