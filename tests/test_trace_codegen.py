import numpy as np
import autogen
from opto.optimizers import OptoPrime
from opto import trace
from opto.trace import node, bundle, GRAPH    
from cristal.llm.codegen.dirichlet_bayes import  dirichlet_bayes_codegen
import os
import litellm
from opto.trace.utils import render_opt_step

"""
Test the codegen validation workflow with trace
"""


def get_feedback(predict, target):
    if np.array_equal(predict, target):
        return "test case passed!"
    else:
        feedback0 = """
        Read the docstring and the prompt description carefully. Check the mathematical formula for the operation in the functions name.
        """
        feedback1 = """
        Expected {}, but got {}. Please check the implementation.
        """.format(target.data, predict.data)
        
        return "test case failed! \n"+feedback0+feedback1


@bundle(trainable=True)
def verifier():
    """
    For a coding problem described below:
    Given list of integers, return list in strange order.
    Strange sorting, is when you start with the minimum value,
    then maximum of the remaining integers, then minimum and so on.

    Return a test input and an expected output to verify if the function is implemented correctly.
    """
    test_input = {
        "likelihood": np.array([[0.1, 0.1, 0.8], [0.1, 0.6, 0.3], [0.5, 0.4, 0.01]]),
        "p_k": np.array([0.2, 0.2, 0.6]),
        "alphas": np.array([1, 1, 1]),
    }
    expected_output = (
        np.array([0.05882353, 0.05882353, 0.88235294]),
        np.array([1.05882353, 1.05882353, 1.88235294]),
    )   
    expected_output = expected_output
    return test_input, expected_output

if __name__ == '__main__':
    SEED = 101010
    GRAPH.clear()
    max_iterations = 100
    test_ground_truth = (
        np.array([0.05882353, 0.05882353, 0.88235294]),
        np.array([1.05882353, 1.05882353, 1.88235294]),
    )   
    test_ground_truth_input = {
        "likelihood": np.array([[0.1, 0.1, 0.8], [0.1, 0.6, 0.3], [0.5, 0.4, 0.01]]),
        "p_k": np.array([0.2, 0.2, 0.6]),
        "alphas": np.array([1, 1, 1]),
    }

    epoch = 2

    config_list = autogen.config_list_from_json("OAI_CONFIG_LIST")
    os.environ["TRACE_LITELLM_MODEL"] = "ollama/qwen2.5-coder:32b"

    #if isinstance(config_list, list) and len(config_list) > 0:
    #    os.environ["OPENAI_API_KEY"] = config_list[0].get("api_key", "")
    #else:
    #    raise ValueError("Invalid configuration list or missing 'OPENAI_API_KEY'")
    
    code_optimizer = OptoPrime(dirichlet_bayes_codegen.parameters(),
                            config_list=autogen.config_list_from_json("OAI_CONFIG_LIST"),
                            ignore_extraction_error=True)
    verifier_optimizer = OptoPrime(verifier.parameters(),
                                config_list=autogen.config_list_from_json("OAI_CONFIG_LIST"),
                                ignore_extraction_error=True)
    #litellm._turn_on_debug()
    for i in range(epoch):
        print(f"Training Epoch {i}")

        # Step 2: Coder optimizes till it passes the verifier
        verifier_passed = False
        code_iterations = 0
        while not verifier_passed:
            print(f"Code function update {code_iterations}")
            code_iterations += 1
            if code_iterations > max_iterations:
                print(f"Reached max iterations at {code_iterations}")
                break
            try:
                test_input, expected_output = verifier()
                test_output = dirichlet_bayes_codegen(**test_input.data)
                # note that we use the same feedback function as before
                feedback = get_feedback(test_output, expected_output)
            except trace.ExecutionError as e:
                feedback = e.exception_node.data
                test_output = e.exception_node
                expected_output = None
                
            verifier_passed = np.array_equal(test_output, expected_output)
            print(f"Verifier passed: {verifier_passed}")
            code_optimizer.zero_feedback()
            code_optimizer.backward(test_output, feedback)

            code_optimizer.step(verbose=True)

        # Step 3: Coder is checked by ground truth
        try:
            test_output = dirichlet_bayes_codegen(**test_ground_truth_input.data)
            feedback = get_feedback(test_output, test_ground_truth)
        except trace.ExecutionError as e:
            feedback = e.exception_node.data
            test_output = e.exception_node
        correctness = np.array_equal(test_output, test_ground_truth)
        
        if correctness:
            break

        # Step 4: If the Coder fails, Verifier needs to propose better unit tests
        print(f"Verifier update")
        feedback = "Verifier proposed wrong test_input, expected_output that do not validate the implementation of the function."
        verifier_optimizer.zero_feedback()
        verifier_optimizer.backward(test_output, feedback, visualize=True)
        verifier_optimizer.step(verbose=True)
    
    print(dirichlet_bayes_codegen.parameters()[0])
    #print(f"Final code: {dirichlet_bayes_codegen.code}")
    #print(f"Final verifier: {verifier.code}")
    #print(f"Final test_output: {test_output}")
    #print(f"Final expected_output: {test_ground_truth}")
    #print(f"Final feedback: {feedback}")
    #print(f"Final correctness: {correctness}")
    #print(f"Final verifier_passed: {verifier_passed}")
    #print(f"Final test_input: {test_input}")
    #print(dirichlet_bayes_codegen.parameters()[0])
    #print(verifier.parameters()[0])