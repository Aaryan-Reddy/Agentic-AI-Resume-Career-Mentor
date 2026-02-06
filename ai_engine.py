import os
from secret_key import OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from tools import extract_skills, build_roadmap, course_finder
from rag_store import rag_query


# ---------------- MODEL (Streaming Enabled) ----------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    streaming=True
)


# ==========================================================
# AGENT 1: ROLE SELECTOR (TOP 3)
# ==========================================================
def role_selector_agent(resume_text, vectorstore, memory):

    plan = "Plan: Read resume → Identify domains → Suggest top 3 roles"

    context = rag_query(vectorstore, "projects skills experience domain")

    prompt = f"""
You are a Role Selector Agent.

{plan}

Resume Evidence:
{context}

Task:
Suggest the TOP 3 job roles this candidate fits best.

Return ONLY role names as numbered list.
"""

    roles_text = llm.invoke([HumanMessage(content=prompt)]).content.strip()

    roles = []
    for line in roles_text.split("\n"):
        if "." in line:
            roles.append(line.split(".")[1].strip())

    memory["best_roles"] = roles[:3]

    return plan, roles[:3]


# ==========================================================
# AGENT 2: CURRENT SKILLS AGENT
# ==========================================================
def current_skills_agent(resume_text, memory):

    plan = "Plan: Extract all current skills from resume"

    skills = extract_skills.invoke({"text": resume_text})
    memory["current_skills"] = skills

    return plan, skills


# ==========================================================
# AGENT 3: ROLE GAP + ROADMAP (RAG GROUNDED)
# ==========================================================
def analyze_roles(memory, vectorstore):

    plan = "Plan: For each role → Retrieve resume evidence → Find missing skills → Roadmap → Courses"

    role_analysis = {}

    for role in memory["best_roles"]:

        # ✅ RAG retrieval for THIS role
        role_context = rag_query(
            vectorstore,
            f"{role} experience projects skills tools"
        )

        gap_prompt = f"""
You are a Skill Gap Agent.

Target Role: {role}

Resume Evidence (Retrieved Context):
{role_context}

Candidate Current Skills:
{memory['current_skills']}

Rules:
- Use resume evidence above (no generic assumptions)
- Do NOT repeat skills candidate already has
- Do NOT include vague skills like "Programming"
- Suggest only real missing skills that strengthen profile
- Limit to 5–7 skills max
- Return ONLY comma-separated missing skills

What are the missing skills for this role?
"""

        missing_text = llm.invoke(
            [HumanMessage(content=gap_prompt)]
        ).content.strip()

        missing_skills = [
            s.strip().lower()
            for s in missing_text.split(",")
            if s.strip()
        ]

        # Roadmap
        roadmap = build_roadmap.invoke(
            {"missing_skills": missing_skills}
        )

        # Courses
        courses = {}
        for skill in missing_skills[:5]:
            courses[skill] = course_finder.invoke({"skill": skill})

        role_analysis[role] = {
            "missing_skills": missing_skills,
            "roadmap": roadmap,
            "courses": courses
        }

    memory["role_analysis"] = role_analysis

    return plan, role_analysis


# ==========================================================
# FINAL REPORT
# ==========================================================
def final_report(memory):

    prompt = f"""
You are a Career Mentor AI.

Candidate Current Skills:
{memory['current_skills']}

Top 3 Best Roles:
{memory['best_roles']}

Role Gap + Roadmap Analysis:
{memory['role_analysis']}

Write a clean structured report:

1. Current Skills Summary
2. Best 3 Roles
3. For each role:
   - Missing Skills (5–7 only)
   - Roadmap
   - Best Courses

Do NOT mention matched skills.
Do NOT repeat current skills in missing skills.
"""

    return llm.invoke([HumanMessage(content=prompt)]).content


# ==========================================================
# FOLLOW-UP CHAT
# ==========================================================
def chat_followup(question, memory, vectorstore):

    context = rag_query(vectorstore, question)

    prompt = f"""
You are a Career Mentor AI.

User Current Skills:
{memory['current_skills']}

Roles:
{memory['best_roles']}

Role Plans:
{memory['role_analysis']}

Resume Evidence:
{context}

User Question:
{question}

Answer clearly and practically.
"""

    return llm.invoke([HumanMessage(content=prompt)]).content
