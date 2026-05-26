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
You are an expert recruiter evaluating candidate-role fit across technical and professional roles.
Evaluate how well this candidate fits this specific role using only the resume and job
description provided below. Do not assume anything that is not supported by the text.

RESUME:
{resume}

JOB TITLE: {title}
JOB DESCRIPTION:
{description}

Score each dimension from 0-25, then sum for a total out of 100. Be selective:
a generally good candidate should not automatically receive a high score unless the
role requirements clearly match the resume.

1. ROLE TYPE MATCH (0-25)
   Does the actual work match the candidate's background and target role?
   Distinguish between different job families and work styles, such as software
   engineering, data/analytics, product, design, research, operations, customer-facing
   roles, leadership, and individual contributor work. Penalize role-type mismatches
   even when some skills overlap.

2. TECHNICAL SKILLS MATCH (0-25)
   Compare required skills separately from preferred or nice-to-have skills.
   Required skills, tools, platforms, methods, certifications, workflows, and
   deliverables should drive most of this score. Preferred skills can strengthen
   an already good match, but they should not compensate for missing must-have
   requirements. Give partial credit for adjacent skills only when they plausibly
   transfer to the role.

3. SENIORITY / SCOPE MATCH (0-25)
   Does the candidate's seniority and expected scope match the role?
   Consider years of experience, ownership level, leadership expectations,
   independence, cross-functional work, people management, and whether the role
   appears junior, mid-level, senior, staff, manager, or executive-level.

4. DOMAIN / INDUSTRY FIT (0-25)
   Compare the candidate's domain background with domain signals in the job
   description, such as industry, customer segment, business model, regulated
   environment, product area, or user type. If the job description gives little
   domain signal, assign a moderate score instead of inventing a domain.

Scoring guidance:
- 21-25: strong direct match with clear evidence
- 16-20: good match with minor gaps
- 10-15: partial or adjacent match
- 5-9: weak match with major gaps
- 0-4: little to no match

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
    "strengths": [
        "<2-4 concrete reasons the resume matches important role requirements>",
        "<focus on specific skills, tools, deliverables, domain knowledge, or role requirements rather than generic work history>"
    ],
    "gaps": [
        "<2-4 concrete missing, weak, unclear, or mismatched areas>",
        "<prioritize missing required skills over minor nice-to-haves; do not mention seniority or experience level>"
    ],
    "reasoning": "<2-3 sentences summarizing the main score drivers, tradeoffs, and why the recommendation follows from the evidence>"
}}

Recommendation thresholds:
- 85-100: "strong apply"
- 65-84: "apply"
- 45-64: "maybe apply"
- 0-44: "don't apply"

Rules:
- The recommendation must match the total_score threshold.
- strengths must be specific, evidence-based, and tied to important role requirements.
- Do not list generic work experience, years of experience, or past job titles as strengths unless they directly prove a required skill, domain fit, or seniority/scope requirement.
- gaps must be specific and should distinguish must-have gaps from preferred-skill gaps.
- Do not use generic statements like "good experience" or "skills match" without naming the actual evidence.
- Use plain text only. No HTML tags, no markdown, no bold formatting.
- Return JSON only. Do not include explanatory text before or after the JSON.\
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
