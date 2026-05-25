from . import database as db, embeddings, job_fetcher, llm_judge


async def run_full_pipeline(top_n: int = 10) -> dict:
    # Step 1: fetch new jobs from Greenhouse
    new_jobs = await job_fetcher.fetch_all_jobs()

    # Step 2: embed all pending jobs and compute cosine similarity vs resume
    ranked = await embeddings.embed_and_rank_pending_jobs()

    # Step 3: re-fetch next batch if we don't have enough ranked jobs
    if db.count_ranked_jobs() == 0 and new_jobs == 0:
        return {
            "new_jobs_fetched": 0,
            "jobs_ranked": 0,
            "jobs_evaluated": 0,
            "message": "No new jobs available. Try adjusting search queries in config.yaml.",
        }

    # Step 4: LLM judge top N
    evaluated = await llm_judge.evaluate_top_jobs(top_n=top_n)

    return {
        "new_jobs_fetched": new_jobs,
        "jobs_ranked": ranked,
        "jobs_evaluated": evaluated,
    }
