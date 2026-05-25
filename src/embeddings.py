import logging
import yaml
import numpy as np
from pathlib import Path
from openai import AsyncOpenAI

from . import database as db

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
client = AsyncOpenAI()


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))


async def get_embedding(text: str, model: str) -> list[float]:
    text = text.replace("\n", " ")
    response = await client.embeddings.create(input=text, model=model)
    return response.data[0].embedding


def _extract_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n".join(
            page.extract_text() for page in reader.pages if page.extract_text()
        )
    return path.read_text()


async def load_resume() -> str:
    """ Load the resume from the path in config.yaml, embed it, and store in DB. """
    config = load_config()
    resume_path = Path(config["candidate"]["resume_path"])
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume not found: {resume_path}")

    content = _extract_text(resume_path)
    model = config["openai"]["embedding_model"]
    embedding = await get_embedding(content, model)
    db.save_resume(content, embedding)
    return content


async def embed_and_rank_pending_jobs() -> int:
    """ Embed all pending jobs and rank them by cosine similarity to the resume. """
    config = load_config()
    model = config["openai"]["embedding_model"]

    resume = db.get_resume()
    if not resume:
        raise ValueError("No resume in DB — run load_resume first.")

    from tqdm.asyncio import tqdm
    pending = db.get_pending_jobs(limit=500)
    logger.info(f"Embedding {len(pending)} pending jobs...")
    for job in tqdm(pending, desc="Embedding jobs", unit="job"):
        text = f"{job['title']}\n{job['company']}\n{job['description']}"
        embedding = await get_embedding(text, model)
        score = cosine_similarity(resume["embedding"], embedding)
        db.update_job_embedding(job["id"], embedding, score)

    logger.info(f"Ranking complete — {len(pending)} jobs ranked")
    return len(pending)
