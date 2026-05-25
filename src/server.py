from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from fastmcp import FastMCP
from . import database as db, embeddings, job_fetcher, llm_judge, pipeline

mcp = FastMCP("Job Application Prioritizer")


@mcp.tool()
async def load_resume() -> str:
    """Load the resume from the path in config.yaml, embed it, and store in DB."""
    content = await embeddings.load_resume()
    return f"Resume loaded and embedded successfully ({len(content)} chars)."


@mcp.tool()
async def fetch_jobs() -> str:
    """Fetch new job listings from Greenhouse based on config.yaml search queries."""
    new_jobs = await job_fetcher.fetch_all_jobs()
    stats = db.get_stats()
    return f"Fetched {new_jobs} new jobs. DB stats: {stats}"


@mcp.tool()
async def embed_and_rank() -> str:
    """Embed all pending jobs and rank them by cosine similarity to the resume."""
    ranked = await embeddings.embed_and_rank_pending_jobs()
    stats = db.get_stats()
    return f"Ranked {ranked} jobs by cosine similarity. DB stats: {stats}"


@mcp.tool()
async def evaluate_jobs(top_n: int = 10) -> str:
    """Run LLM-as-Judge on the top N ranked jobs and store apply/skip recommendations."""
    evaluated = await llm_judge.evaluate_top_jobs(top_n=top_n)
    return f"LLM evaluated {evaluated} jobs."


@mcp.tool()
async def run_pipeline(top_n: int = 10) -> dict:
    """Run the full pipeline: fetch → embed & rank → LLM judge. One call does it all."""
    return await pipeline.run_full_pipeline(top_n=top_n)


@mcp.tool()
async def get_recommendations() -> list[dict]:
    """Return all jobs recommended to apply for, sorted by LLM score (highest first)."""
    jobs = db.get_evaluated_jobs()
    return [
        {
            "id": j["id"],
            "title": j["title"],
            "company": j["company"],
            "location": j["location"],
            "llm_score": j["llm_score"],
            "reasoning": j["llm_reasoning"],
            "strengths": j["strengths"],
            "gaps": j["gaps"],
            "url": j["url"],
        }
        for j in jobs
    ]


@mcp.tool()
async def get_all_evaluations() -> list[dict]:
    """
    Return all evaluated jobs sorted by score (highest first).

    Group by recommendation in this order: 💚 Strong Apply, 🟡 Apply, 🟠 Maybe Apply, ❌ Don't Apply.

    For each job, use bullet points — NOT a table, NOT inline text:
    • **{title} — {company}** · {score}/100
    • 📍 {location}
    • Apply: {url}
    • ✅ Strengths: {strength1} / {strength2}
    • ❌ Gaps: {gap1} / {gap2}
    """
    jobs = db.get_evaluated_jobs()
    return [
        {
            "title": j["title"],
            "company": j["company"],
            "score": j["llm_score"],
            "recommendation": j["recommendation"],
            "location": j["location"],
            "url": j["url"],
            "strengths": j["strengths"],
            "gaps": j["gaps"],
        }
        for j in jobs
    ]



@mcp.tool()
def dismiss_batch() -> str:
    """
    Call this when the user says they are done with the current batch
    (e.g. 'I've applied to these', 'done with this batch', 'show me new ones').
    Marks all evaluated jobs as dismissed so they won't appear in future results.
    The next evaluate_jobs() call will automatically start from the next ranked batch.
    """
    count = db.dismiss_current_batch()
    remaining = db.count_ranked_jobs()
    return f"Dismissed {count} jobs. {remaining} ranked jobs still waiting to be evaluated."


@mcp.tool()
def reset_database() -> str:
    """
    Reset the database — deletes all jobs and starts fresh.
    Use when the user wants to start over or clear all job data.
    Resume embedding is kept.
    """
    with db.get_connection() as conn:
        conn.execute("DELETE FROM jobs")
    return "Database cleared. All jobs deleted, resume kept. Run run_pipeline() to start fresh."


@mcp.tool()
def get_stats() -> dict:
    """Return a count of jobs by status (pending, ranked, evaluated, dismissed, applied)."""
    return db.get_stats()


def main():
    db.init_db()
    mcp.run()


if __name__ == "__main__":
    main()
