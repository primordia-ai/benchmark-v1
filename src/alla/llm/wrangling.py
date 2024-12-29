"""
Tools and objects for information wrangling and densification

Below, the SummaryInit class description and its field descriptions are actually used
to inform langauge models.

See: https://python.useinstructor.com/tutorials/6-chain-of-density/#data-classes
for more info on the Chain-of-Density used here.
"""
import os
from typing import TYPE_CHECKING

import instructor
import nltk
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

from alla.utils import PROJECT_ROOT

if TYPE_CHECKING:
    from alla.llm.base import LLMBaseClass, logger


class SummaryInit(BaseModel):
    """
    This is an initial summary which should be long (4-5 sentences, ~80 words) yet
    highly non-specific, containing little information beyond the entities marked as missing.
    Use overly verbose languages and fillers (Eg. This document discusses) to reach ~80 words.
    """

    summary: str = Field(
        ...,
        description="This is a summary of the document provided which is overly verbose and uses fillers. "
                    "It should be roughly 80 words in length",
    )


class RewrittenSummary(BaseModel):
    """
    This is a new, denser summary of identical length which covers every entity and detail 
    from the previous summary plus the Missing Entities. Guidelines 
        - Make every word count: Rewrite the previous summary to improve flow and make space for additional entities 
        - Never drop entities from the previous summary. If space cannot be made, add fewer new entities. 
        - The new summary should be highly dense and concise yet self-contained, 
            eg., easily understood without the Document.
        - Make space with fusion, compression, and removal of uninformative phrases like "the document discusses"
        - Missing entities can appear anywhere in the new summary 
        
    An Entity is a real-world object that's assigned a name. For example, a person, country, a product, etc.
    """
    summary: str = Field(
        ...,
        description="This is a new, denser summary which covers every entity and detail from the previous "
                    "summary plus the Missing Entities. It should have the same length (~80 words) as the "
                    "previous summary and should be easily understood without the Document",
    )
    absent: list[str] = Field(
        ...,
        default_factory=list,
        description="This is a list of Entities found absent from the new summary that were present "
                    "in the previous summary",
    )
    missing: list[str] = Field(
        default_factory=list,
        description="This is a list of 1-3 informative Entities from the Document that are missing "
                    "from the new summary which should be included in the next generated summary.",
    )

    @field_validator("summary")
    def min_length(cls, v: str):
        tokens = nltk.word_tokenize(v)
        num_tokens = len(tokens)
        if num_tokens < 60:
            raise ValueError(
                "The current summary is too short. Please make sure that you generate a new summary that is "
                "around 80 words long."
            )
        return v

    @field_validator("missing")
    def has_missing_entities(cls, missing_entities: list[str]):
        if len(missing_entities) == 0:
            raise ValueError(
                "You must identify 1-3 informative Entities from the Document which are missing from "
                "the previously generated summary to be used in a new summary"
            )
        return missing_entities

    @field_validator("absent")
    def has_no_absent_entities(cls, absent_entities: list[str]):
        absent_entity_string = ",".join(absent_entities)
        if len(absent_entities) > 0:
            print(f"Detected absent entities of {absent_entity_string}")
            raise ValueError(
                f"Do not omit the following Entities {absent_entity_string} from the new summary"
            )
        return absent_entities


def summarize_document(llm: "LLMBaseClass", document: str, summary_steps: int = 3):

    client = instructor.from_openai(
        OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # required, but unused
        ),
        mode=instructor.Mode.JSON,
    )

    summary_chain = []  # We first generate an initial summary
    summary_init: SummaryInit = client.chat.completions.create(
        model=llm.code,
        response_model=SummaryInit,
        messages=[
            {
                "role": "system",
                "content": "Write a summary about the document that is long (4-5 sentences) yet highly non-specific. "
                           "Use overly, verbose language and fillers(eg.,'this document discusses') to reach ~80 words",
            },
            {
                "role": "user", "content": f"Here is the Document: {document}"
            },
            {
                "role": "user", "content": "The generated summary should be about 80 words."
            },
        ], max_retries=2,
    )

    summary_chain.append(summary_init)
    missing_entity_messages = ([])
    prev_summary = None
    for _i in range(summary_steps):
        new_summary: RewrittenSummary = client.chat.completions.create(
            model=llm.code,
            messages=[
                {
                    "role": "system",
                    "content": """
                    You are going to generate an increasingly concise, entity-dense summary of the following document. 
                    Perform the following two tasks - Identify 1-3 informative entities from the following document 
                    which is missing from the previous summary 
                        - Write a new denser summary of identical length which covers every entity and detail from 
                        the previous summary plus the Missing Entities Guidelines. 
                        - Make every word count: re-write the previous summary to improve flow and make space for 
                        additional entities. 
                        - Make space with fusion, compression, and removal of uninformative phrases 
                        like "the document discusses". 
                        - The summaries should become highly dense and concise yet self-contained, 
                        e.g., easily understood without the Document. 
                        - Missing entities can appear anywhere in the new summary. 
                        - Never drop entities from the previous summary. If space cannot be made, add fewer new entities. 
                    """,
                }, {
                    "role": "user", "content": f"Here is the Document: {document}"
                },
                {
                    "role": "user", "content": f"Here is the previous summary: {prev_summary}",
                },
                *missing_entity_messages,
            ],
            max_retries=3,
            max_tokens=1000,
            response_model=RewrittenSummary,
        )
        summary_chain.append(new_summary.summary)
        # For next iter
        missing_entity_messages = ([
            {
                "role": "user",
                "content": f"Please include these Missing Entities: {','.join(new_summary.missing)}",
            },
        ])
        prev_summary = new_summary

    return summary_chain


def read_reports(
    report_dir: str = os.path.join(PROJECT_ROOT, "reports"),
    include: list = None,
    exclude: list = None,
):
    if not os.path.exists(report_dir):
        raise FileNotFoundError(f"Report directory {report_dir} not found")

    # Read the documents for the asset, with recursion if no reports list included
    reports = ""

    for filename in os.listdir(report_dir):
        if exclude is not None and filename in exclude:
            continue
        if include is not None and filename not in include:
            continue

        file_path = os.path.join(report_dir, filename)
        logger.debug(f"Reading file {file_path}")
        with open(file_path, 'r') as file:
            reports += "\n<START DOCUMENT>" + file.read() + "\n<END DOCUMENT>"

    return reports
