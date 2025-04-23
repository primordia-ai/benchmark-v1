"""
Experiment 3

Through iterations on an actor-critic loop, can powerful LLMs arrive at correct program
solutions to our inference problem? And if so, how long does this take?
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
from cristal.llm.codegen.likelihood_inference import IndicatorLikelihoodInference
from cristal.logger import capture_description, log_to_file, logger, reset_logger


# Define different task difficulties based on how much information we will provide
TASK_HARD = """
We are trying to categorize a company into one of K classes and we observe J categorical indicators.  
Each indicator can take one of C colors, that characterize the company's value in the dimension of that indicator.
Each indicator has a different set of likelihoods given classes. We represent the sampling distributions as

k ~ Categorical(p)
x_j ~ Categorical(A_{j,:,k}) # The array A (J x C x K) contains the likelihoods of each indicator given each of the latent company class k.

The goal is to update an indicator likelihood prior using Bayesian inference, from N observations of indicator colors and 
categories. Each indicator has a different set of likelihoods given classes.
"""

TASK_MEDIUM = TASK_HARD + """

Tip:
    First work out the closed-form Bayesian update for the class probability vector p.
    Then extend this to assume the likelihood matrix is also unknown and we must simultaneously infer its parameters.
"""

TASK_EASY = TASK_MEDIUM + """

Recall that the unnormalized Bayesian posterior is:
posterior = likelihood * prior
where the likelihood is the indicator likelihood given the class and the prior is the prior distribution of classes.

Your solution may resemble the following:

posterior = prior.copy()
n_cat = prior.shape[0]
J = prior.shape[1]
n_color = prior.shape[2]

for k_val in range(n_cat):
    for j in range(J):  # Iterate over indicators
        for c in range(n_color):  # Iterate over colors
            posterior[k_val, j, c] = prior[k_val, j, c] ...
epsilon = 1e-10  # Small value to prevent division by zero
normalized = posterior / (np.sum(posterior, axis=2, keepdims=True) + epsilon)
return normalized
"""


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
        "--compress-reports", action="store_true", help="Attempt to compress the reports through recursive summarizing"
    )
    parser.add_argument(
        "--reports-dir", type=str, default=None, help="Reports directory override over env REPORTS_DIR."
    )

    args = parser.parse_args()

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
        # Define the task
        task = TASK_EASY
        # Set retries and budget
        actor_retries, actor_budget = 3, 20
        loop_max_iters = 4
        loop_budget = args.budget
        # Set logger to debug

    else:
        # Define the task
        task = TASK_HARD
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
    validator = IndicatorLikelihoodInference

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
        "task": task,
    }

    config_path = os.path.join(results_dir, "config.json")
    with open(config_path, 'w') as f:
        logger.info(f"Saving run data to {config_path}")
        json.dump(run_data, f, indent=4)

    results = {}
    # Call the loop method
    func = actor.critic_loop(
        validator, loop_max_iters, loop_budget, actor_retries, actor_budget
    )

    # Save Results
    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, 'w') as f:
        logger.info(f"Saving results to {results_path}")
        json.dump(results, f, indent=4)


if __name__ == '__main__':

    pytensor.config.exception_verbosity = 'high'

    # Set seed
    # 101010 happens to be 42 in binary. How charming.
    SEED = 101010

    main()
