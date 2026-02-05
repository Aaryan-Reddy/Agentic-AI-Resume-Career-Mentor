# Agentic AI Resume Career Mentor

An agentic AI application that analyzes a candidate’s resume and provides career guidance using Retrieval-Augmented Generation (RAG), a vector database, and multi-agent reasoning. The system identifies the candidate’s current skills, suggests the best-fit job roles, highlights real skill gaps, and generates personalized learning roadmaps with course recommendations and conversational follow-up support.

---

## Key Features

- Resume analysis directly from PDF  
- Identifies the top 3 best-fit career roles for the candidate  
- Extracts current skills from the resume using an LLM-based skill parser  
- Performs role-specific skill gap analysis using grounded multi-agent reasoning  
- Generates a roadmap for missing skills (5–7 key gaps per role)  
- Course recommendations using real-time Google search via SerpAPI  
- Retrieval-Augmented Generation (RAG) with FAISS vector database  
- Multi-agent workflow with visible planning + execution steps  
- Chat-based follow-up interaction after the initial report  
- Session memory for contextual career mentoring  

---

## Architecture Overview

High-level flow:

1. Resume PDF is parsed and indexed into a FAISS vector database  
2. Resume context is retrieved using RAG for grounding responses  
3. A Role Selector Agent suggests the top 3 career roles  
4. A Skills Agent extracts current skills from the full resume  
5. A Skill Gap Agent identifies true missing skills per role (LLM-driven, not keyword subtraction)  
6. A Roadmap Coach Agent builds a learning plan and fetches courses using SerpAPI  
7. An Interview Prep Agent generates preparation strategy and questions  
8. Users can ask follow-up questions through an interactive chat interface  

---

## Core Concepts Used

- Vector Databases (FAISS)  
- Retrieval-Augmented Generation (RAG)  
- Multi-Agent Role-Based Reasoning  
- Agent Planning and Execution Traces  
- Tool Calling with LangChain  
- Session-Based Conversational Memory  
- Real-Time Learning Resource Search (SerpAPI)  

---

## Tech Stack

- Python  
- LangChain  
- OpenAI API  
- FAISS  
- Streamlit  
- PyPDFLoader  
- SerpAPI (Google Search Integration)  

---

## Project Structure

```text
career_mentor/
├── app.py          # Streamlit frontend + chat UI
├── ai_engine.py    # Multi-agent reasoning pipeline
├── tools.py        # Tools: skill extraction, roadmap, course search
├── rag_store.py    # FAISS vector DB + RAG retrieval
├── memory.py       # Session memory storage
├── secret_key.py   # API keys (not committed)
