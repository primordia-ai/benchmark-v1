"""
Codegen Critic module
"""

from instructor.exceptions import InstructorRetryException

from cristal.llm.base import LLMBaseClass
from cristal.llm.codegen.validator import LLMCodeGenException
from cristal.llm.utils import get_retry_control


class CodeGenCritic:
    """
    CodeGen Critic.

    The CodeGen Critic analyzes codegen responses from an LLMCodeGen (Agent) model, and provides tailored
    guidance from its own (typically superior) LLM.
    Both the Critic and the Agent are given the objective/task.
    """

    def __init__(self, llm: LLMBaseClass, task: str):
        self.llm: LLMBaseClass = llm
        self.task: str = task
        # Store a history of the agent solutions and the critic responses
        # Keys of dict items will be "code", "evaluation", and "approved"
        self.history: list[dict] = []

    def evaluate_function(
        self,
        code: str,
        max_retries: int = 3,
        budget: float = 30,  # seconds
    ) -> tuple[str, bool]:
        """
        Evaluate the proposed function code and return a text response, and an approval boolean.

        :param code:            The proposed function code.
        :param max_retries:     Maximum number of retries to attempt.
        :param budget:          Budget to perform the evaluation within.
        :return:                Text response and approval boolean.
        """
        client = self.llm.get_client()
        if "gemini" in self.llm.code.lower():
            additional_kwargs = {}
        else:
            additional_kwargs = {
                "model": self.llm.code,
                "temperature": self.llm.temperature,
                "seed": self.llm.seed,
            }

        retry_ctrl = get_retry_control(max_retries, budget, wait_range=(0.1, 0.3))
        prompt = self._get_eval_prompt(code)

        try:
            response, completion = client.chat.completions.create_with_completion(
                **additional_kwargs,
                messages=[
                    {
                        "role": "developer",  # OpenAI docs say this over 'system' for newer models. Hopefully is
                        # back-compatible
                        "content": self.llm.system_roles["codegen_critic"],
                    },
                    {
                        "role": "user",
                        "content": prompt
                    },
                ],
                **retry_ctrl
            )
        except InstructorRetryException as e:
            raise LLMCodeGenException(f"Error: {e}")

        return response, completion

    def _get_prompt(self, code: str) -> str:
        """
        Get the prompt for the critic.

        :param code:    The code to be evaluated.
        :return:        The prompt string.
        """
        prompt: str = f"""
        Given the task described, please evaluate the proposed code.
        Respond with feedback on correctness, issues, and improvements.
        The task is given now, enclosed in <TASK> html tags.
        
        <TASK>
        {self.task}
        </TASK>
        """

        if self.history:
            prompt += f"""
            
            There exists a previous attempt and your associated evaluation.
            These are delimited by <PREV_SOLN> and <PREV_EVAL> tags.

            <PREV_SOLN>            
            {self.history['code']}
            </PREV_SOLN>
            
            <PREV_EVAL>
            {self.history['evaluation']}
            </PREV_EVAL>
            """

        prompt += f"""
        The code is given now, enclosed in <SOLN> html tags.

        <SOLN>
        {code}
        </SOLN>

        Please provide your evaluation.
        Format your response in JSON format with the following keys:
        - 'evaluation' (str): Your evaluation of the code.
        - 'approved' (bool): Boolean approval of the code.
        """

        return prompt
