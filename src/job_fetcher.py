import re
import logging
from datetime import datetime, timezone, timedelta
import httpx
import yaml
from pathlib import Path

from . import database as db

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
GREENHOUSE_BASE = "https://boards-api.greenhouse.io/v1/boards"
US_LOCATION_KEYWORDS = ["united states", "usa", "u.s.", "us"]


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _strip_html(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()


def _title_matches(title: str, keywords: list[str]) -> bool:
    """ Pull the job if its title matches any of the keywords """
    title_lower = title.lower()
    return any(kw.lower() in title_lower for kw in keywords)


def _title_excluded(title: str, exclude_keywords: list[str]) -> bool:
    title_lower = title.lower()
    return any(kw.lower() in title_lower for kw in exclude_keywords)


def _keyword_in_text(text: str, keyword: str) -> bool:
    pattern = rf"(?<![a-z0-9]){re.escape(keyword.lower())}(?![a-z0-9])"
    return re.search(pattern, text) is not None


def _is_allowed_location(location: str, location_filter: dict) -> bool:
    if not location_filter.get("us_only", False):
        return True

    loc = location.strip().lower()
    is_us = any(_keyword_in_text(loc, keyword) for keyword in US_LOCATION_KEYWORDS)
    is_plain_remote = location_filter.get("allow_plain_remote", False) and loc == "remote"

    if not (is_us or is_plain_remote):
        return False

    cities = location_filter.get("cities", [])
    if cities and not is_plain_remote:
        return any(_keyword_in_text(loc, city) for city in cities)

    return True


def _extract_min_years(description: str) -> int | None:
    """Extract the years of experience required from a job description."""
    patterns = [
        r"(\d+)\+\s*years?\s*(?:of\s*)?experience",
        r"(\d+)\s*(?:to|-)\s*\d+\s*years?\s*(?:of\s*)?experience",
        r"minimum\s*(?:of\s*)?(\d+)\s*years?",
        r"at least\s*(\d+)\s*years?",
    ]
    for pattern in patterns:
        match = re.search(pattern, description.lower())
        if match:
            return int(match.group(1))
    return None


async def _fetch_company_jobs(
    token: str, company_name: str, keywords: list[str],
    exclude_keywords: list[str], max_years: int, min_years_floor: int,
    cutoff: datetime | None, location_filter: dict,
) -> list[dict]:
    url = f"{GREENHOUSE_BASE}/{token}/jobs"

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(url, params={"content": "true"})
        if response.status_code == 404:
            return []
        response.raise_for_status()
        data = response.json()

    jobs = []
    for item in data.get("jobs", []):
        title = item.get("title", "")
        if not _title_matches(title, keywords):
            continue
        if _title_excluded(title, exclude_keywords):
            continue

        location = item.get("location", {}).get("name", "")
        if not _is_allowed_location(location, location_filter):
            continue

        updated_at = item.get("updated_at")
        if updated_at and cutoff:
            job_date = datetime.fromisoformat(updated_at)
            if job_date.tzinfo is None:
                job_date = job_date.replace(tzinfo=timezone.utc)
            if job_date < cutoff:
                continue

        description = _strip_html(item.get("content", ""))
        min_years = _extract_min_years(description)
        if min_years is not None and (min_years > max_years or min_years < min_years_floor):
            continue

        jobs.append(
            {
                "id": f"gh_{token}_{item['id']}",
                "title": title,
                "company": company_name,
                "location": location,
                "description": description,
                "url": item.get("absolute_url", ""),
                "salary_min": None,
                "salary_max": None,
                "posted_at": item.get("updated_at"),
            }
        )

    return jobs


async def fetch_all_jobs() -> int:
    config = load_config()
    keywords = config["candidate"]["job_titles"]
    exclude_keywords = config["candidate"].get("exclude_title_keywords", [])
    max_years = config["candidate"].get("max_required_years", 99)
    min_years_floor = config["candidate"].get("min_required_years", 0)
    location_filter = config["candidate"].get("location_filter", {})
    companies = config["greenhouse"]["companies"]

    posted_within_days = config["greenhouse"].get("posted_within_days")
    cutoff = (datetime.now(timezone.utc) - timedelta(days=posted_within_days)) if posted_within_days else None

    new_count = 0
    for entry in companies:
        token = entry["token"]
        name = entry["name"]
        jobs = await _fetch_company_jobs(
            token, name, keywords, exclude_keywords, max_years,
            min_years_floor, cutoff, location_filter
        )
        company_new = 0
        for job in jobs:
            if not db.job_exists(job["id"]):
                db.insert_job(job)
                company_new += 1
        logger.info(f"[{name}] matched {len(jobs)} jobs, {company_new} new")
        new_count += company_new

    logger.info(f"Fetch complete — {new_count} new jobs total")
    return new_count
