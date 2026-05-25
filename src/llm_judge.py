import re
import json
import logging
import yaml
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, field_validator, model_validator
from openai import AsyncOpenAI

from . import database as db

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
client = AsyncOpenAI()


class DimensionScores(BaseModel):
    role_type_match: int
    technical_skills_match: int
    experience_level_match: int
    domain_fit: int


class JudgeResult(BaseModel):
    scores: DimensionScores
    total_score: int
    recommendation: Literal["strong apply", "apply", "maybe apply", "don't apply"]
    strengths: list[str]
    gaps: list[str]
    reasoning: str

    @field_validator("recommendation", mode="before")
    @classmethod
    def normalize_recommendation(cls, v: str) -> str:
        # Strip HTML tags e.g. "<strong>strong apply</strong>" → "strong apply"
        v = re.sub(r"<[^>]+>", "", str(v)).strip().lower()
        # Normalize common variants
        if "strong" in v:
            return "strong apply"
        if "don" in v or "not" in v or "skip" in v:
            return "don't apply"
        if "maybe" in v:
            return "maybe apply"
        if "apply" in v:
            return "apply"
        return "don't apply"

    @model_validator(mode="after")
    def check_total_score(self) -> "JudgeResult":
        expected = sum([
            self.scores.role_type_match,
            self.scores.technical_skills_match,
            self.scores.experience_level_match,
            self.scores.domain_fit,
        ])
        if abs(self.total_score - expected) > 2:
            self.total_score = expected
        return self

JUDGE_PROMPT = """\
You are an expert technical recruiter specializing in data science hiring. \
Evaluate how well this candidate fits this specific role across 4 dimensions.

RESUME:
{resume}

JOB TITLE: {title}
JOB DESCRIPTION:
{description}

Score each dimension from 0-25, then sum for a total out of 100:

1. ROLE TYPE MATCH (0-25)
   Does the nature of the work match? A business-focused DS (causal inference, A/B testing,
   stakeholder reporting) is NOT a good fit for an ML engineering role (model serving,
   MLOps, deep learning research), and vice versa. Be strict here.

2. TECHNICAL SKILLS MATCH (0-25)
   Do the specific tools, methods, and tech stack overlap?
   Partial overlap = partial credit. No overlap = 0.

3. EXPERIENCE LEVEL MATCH (0-25)
   Does the candidate's seniority and years of experience fit what the role requires?
   Overqualified or underqualified both lose points.

4. DOMAIN / INDUSTRY FIT (0-25)
   Does the candidate's background domain (e.g. e-commerce, finance, healthcare)
   align with the company's domain? Adjacent = partial credit.

Respond ONLY with valid JSON in this exact format:
{{
    "scores": {{
        "role_type_match": <0-25>,
        "technical_skills_match": <0-25>,
        "experience_level_match": <0-25>,
        "domain_fit": <0-25>
    }},
    "total_score": <0-100>,
    "recommendation": "<strong apply | apply | maybe apply | don't apply>",
    "strengths": ["<specific strength1>", "<specific strength2>"],
    "gaps": ["<specific gap1>", "<specific gap2>"],
    "reasoning": "<2-3 sentences explaining the key factors>"
}}

Recommendation thresholds:
- 85-100 → "strong apply"
- 65-84  → "apply"
- 45-64  → "maybe apply"
- 0-44   → "don't apply"

IMPORTANT: Use plain text only. No HTML tags, no markdown, no bold formatting.\
"""


async def _judge_single(job: dict, resume_content: str, model: str) -> JudgeResult:
    '''
    Evaluate a single job against the resume using the LLM-as-Judge.
    Args:
        job: dict - the job to evaluate
        resume_content: str - the resume content
        model: str - the model to use

    Returns:
        JudgeResult - the result of the evaluation
    '''

    prompt = JUDGE_PROMPT.format(
        resume=resume_content,
        title=job["title"],
        description=job["description"],
    )
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    return JudgeResult.model_validate(json.loads(response.choices[0].message.content))


async def evaluate_top_jobs(top_n: int = 10) -> int:
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    model = config["openai"]["judge_model"]

    resume = db.get_resume()
    if not resume:
        raise ValueError("No resume in DB — run load_resume first.")

    top_jobs = db.get_ranked_jobs(limit=top_n)
    if not top_jobs:
        return 0

    logger.info(f"LLM judging top {len(top_jobs)} jobs...")
    for i, job in enumerate(top_jobs, 1):
        result = await _judge_single(job, resume["content"], model)
        db.update_job_llm(
            job_id=job["id"],
            score=result.total_score,
            recommendation=result.recommendation,
            reasoning=result.reasoning,
            strengths=result.strengths,
            gaps=result.gaps,
        )
        logger.info(f"  [{i}/{len(top_jobs)}] {result.recommendation.upper()} {result.total_score}/100 — {job['company']} - {job['title']}")

    logger.info("LLM judge complete")
    return len(top_jobs)
