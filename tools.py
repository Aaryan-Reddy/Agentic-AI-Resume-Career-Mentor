import requests
from langchain_core.tools import tool
from secret_key import SERPAPI_API_KEY

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


# Skill extractor LLM
skill_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ---------------- CLEAN SKILL EXTRACTION ----------------
@tool
def extract_skills(text: str) -> list:
    """
    Extract ONLY skill names from resume.
    No proficiency words, no explanations.
    """

    prompt = f"""
Extract ONLY the skill names from this resume.

Rules:
- Do NOT add proficiency text
- Do NOT explain anything
- Output must be comma-separated skill names only

Resume:
{text}
"""

    response = skill_llm.invoke([HumanMessage(content=prompt)]).content

    skills = [s.strip().lower() for s in response.split(",") if s.strip()]
    return sorted(list(set(skills)))


# ---------------- ROADMAP BUILDER ----------------
@tool
def build_roadmap(missing_skills: list) -> list:
    """
    Build roadmap steps (max 7 skills).
    """

    roadmap = []
    for skill in missing_skills[:7]:
        roadmap.append(
            f"Learn {skill} → Build 1 mini project → Add to GitHub"
        )

    return roadmap


# ---------------- REAL COURSE SEARCH TOOL ----------------
@tool
def course_finder(skill: str) -> list:
    """
    Real Google course search using SerpAPI.
    Returns top 3 learning links.
    """

    url = "https://serpapi.com/search.json"

    params = {
        "engine": "google",
        "q": f"best beginner course for {skill}",
        "api_key": SERPAPI_API_KEY
    }

    res = requests.get(url, params=params).json()

    results = []
    for item in res.get("organic_results", [])[:3]:
        results.append({
            "title": item.get("title"),
            "link": item.get("link")
        })

    return results