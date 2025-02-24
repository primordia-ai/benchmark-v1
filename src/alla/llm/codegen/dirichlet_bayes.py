import numpy as np
from pydantic import model_validator, ValidationError

from alla.llm.codegen.base import LLMCodeGen, LLMCodeGenException


class DirichletBayesFunction(LLMCodeGen):
    """
    Validator for the Dirichlet bayesian update function codegen
    """
    _is_function = True
    _function_name = "dirichlet_bayes"
    _required_args = tuple(["likelihood", "p_k", "alphas"])

    @model_validator(mode="after")
    def test_array_shapes(self):
        """
        Validate that the shapes of the arrays are correct
        """
        # Get the function
        exec(self.code)
        dirichlet_bayes = locals()[self._function_name]

        color_pred = np.eye(3)[0]
        # Test the function -- likelihood rows add to 1
        likelihood = np.asarray(
            [
                [0.1, 0.1, 0.8],
                [0.1, 0.6, 0.3],
                [0.5, 0.4, 0.01],
            ]
        )
        likelihood = likelihood @ color_pred.reshape(-1, 1)
        likelihood = likelihood.reshape(3, )
        p_k = np.array([0.2, 0.2, 0.6])
        alphas = np.array([1, 1, 1])

        soln_pk = np.asarray([0.05882353, 0.05882353, 0.88235294])
        soln_alphas = np.asarray([1.05882353, 1.05882353, 1.88235294])
        eps = 1e-6

        try:
            new_pk, new_alphas = dirichlet_bayes(likelihood, p_k, alphas)
        except Exception as e:
            raise LLMCodeGenException(
                "Exception occurred while testing method for test_array_shapes",
                exception=e,
            )
        else:
            # Check we are within error
            if not np.allclose(new_pk, soln_pk, atol=eps):
                raise LLMCodeGenException(
                    "Failed test_array_shapes: new_pk not within tolerance",
                    exception=ValidationError(),
                )

        return self.code


# For testing
_solution_string = '''
def dirichlet_bayes(likelihood: np.ndarray, p_k: np.ndarray, alphas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Bayesian update of the asset's class prediction.
    Does not modify p_k and alphas in place, returns updated versions.

    :param likelihood:      Likelihood matrix for the indicator
    :param p_k:             Class predictions
    :param alphas:          Dirichlet prior alpha parameters
    :return:                Updated class prediction posterior, and alphas
    """
    ###
    # MAP method.
    # From some brief experimenting, is essentially same outcome but very slightly worse variance
    # alphas_star = alphas + posterior - 1
    # alphas += alphas_star / np.sum(alphas_star)
    ###
    posterior = p_k * likelihood
    posterior /= np.sum(posterior)
    # Avoiding modifying in place, just in case
    _alphas = alphas + posterior
    return posterior, _alphas
'''