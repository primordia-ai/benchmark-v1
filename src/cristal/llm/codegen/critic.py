"""
Codegen Critic module
"""
from typing import TypedDict

from instructor.exceptions import InstructorRetryException
from pydantic import BaseModel, Field

from cristal.llm.base import LLMBaseClass
from cristal.llm.codegen.validator import LLMCodeGenException
from cristal.llm.utils import get_retry_control
from cristal.logger import logger


class CriticResponse(BaseModel):
    """
    Critic response model
    """
    evaluation: str = Field(
        ..., title="Evaluation",
        description="The Critic's evaluation of the proposed function solution to the Task."
    )
    approved: bool = Field(
        title="Approved",
        description="Approval boolean of the proposed function solution."
    )


class CriticEvalItem(TypedDict):
    code: str
    evaluation: str
    approved: bool


class DirectiveResponse(BaseModel):
    directive: str


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
        self.history: list[CriticEvalItem] = []

    def reset(self, task: str):
        """
        Reset the critic with a new task.

        :param task:    The new task to set.
        """
        if not task:
            raise ValueError("Task cannot be empty to reset CodeGenCritic.")

        self.task = task
        self.history = []

    def initial_directive_prompt(self) -> str:
        """
        Generate an initial directive that will help the actor generate the Python code solution

        :return:    Directive.
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

        prompt = f"""
        Before we begin evaluating code solutions to the task, please generate a prompt that will help
        the LLM actor model in thinking about its solution.
        
        Try to keep the directive length at most a page of text.
        
        NOTE: The point here is not to create pseudocode for it to fill in the blanks, but more high-level 
        considerations.
        
        Please return your response in JSON format with the single key 'directive'.
        """
        try:
            response, completion = client.chat.completions.create_with_completion(
                **additional_kwargs,
                messages=[
                    {
                        "role": "developer",
                        "content": self.llm.system_roles["codegen_critic"],
                    },
                    {
                        "role": "user",
                        "content": prompt
                    },
                ],
                max_retries=3,
                response_model=DirectiveResponse,
            )
        except InstructorRetryException as e:
            raise LLMCodeGenException(f"Error: {e}")

        return response.model_dump()["directive"]

    def evaluate_function(
        self,
        code: str,
        max_retries: int = 3,
        budget: float = 30,  # seconds
    ) -> CriticEvalItem:
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
                max_retries=retry_ctrl,
                response_model=CriticResponse,
            )
        except InstructorRetryException as e:
            raise LLMCodeGenException(f"Error: {e}")

        result = response.model_dump()
        # Update the history
        logger.debug(f"Critic Eval result:\nAPPROVED: {result['approved']}\nEVALUATION: {result['evaluation']}")
        self.history.append(
            {
                "code": code,
                "evaluation": result["evaluation"],
                "approved": bool(result["approved"]),
            }
        )
        # Return the evaluation
        return self.history[-1]

    def _get_eval_prompt(self, code: str) -> str:
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
            latest = self.history[-1]
            prompt += f"""
            
            There exists an entry from your last evaluation.
            The details will be provided now, with the earlier code attempt within <PREV_SOLN> tags, 
            and your previous evaluation in <PREV_EVAL> tags.

            <PREV_SOLN>            
            {latest['code']}
            </PREV_SOLN>
            
            <PREV_EVAL>
            {latest['evaluation']}
            </PREV_EVAL>
            """

        prompt += f"""
        The proposed solution is given now, enclosed in <SOLN> html tags.

        <SOLN>
        {code}
        </SOLN>

        Please provide your evaluation.
        Format your response in JSON format with the following keys:
        - 'evaluation' (str): Your evaluation of the code.
        - 'approved' (bool): Boolean approval of the code.
        """

        return prompt
