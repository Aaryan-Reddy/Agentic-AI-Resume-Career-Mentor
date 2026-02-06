import streamlit as st
import tempfile

from rag_store import load_pdf, build_vector_store
from memory import init_memory

from ai_engine import (
    role_selector_agent,
    current_skills_agent,
    analyze_roles,
    final_report,
    chat_followup
)

st.set_page_config(page_title="AI Resume Mentor")
st.title("Multi-Agent Resume Career Mentor")


# ---------------- INIT ----------------
if "memory" not in st.session_state:
    st.session_state.memory = init_memory()
    st.session_state.ready = False

if "chat" not in st.session_state:
    st.session_state.chat = []


# ---------------- UPLOAD RESUME ----------------
if not st.session_state.ready:

    resume = st.file_uploader("Upload Resume PDF", type=["pdf"])

    if st.button("Analyze Resume"):

        with st.spinner("Running Career Mentor Agents..."):

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                f.write(resume.read())
                docs = load_pdf(f.name)

            # ✅ Full resume text
            resume_text = "\n".join([d.page_content for d in docs])

            vectorstore = build_vector_store(docs)

            st.session_state.vectorstore = vectorstore
            st.session_state.memory["resume_text"] = resume_text

            # ---------- AGENT PIPELINE ----------
            p1, roles = role_selector_agent(
                resume_text,
                vectorstore,
                st.session_state.memory
            )
            st.markdown(f"Role Selector Agent\n\n*{p1}*\n\n✅ Roles: {roles}")

            p2, skills = current_skills_agent(
                resume_text,
                st.session_state.memory
            )
            st.markdown(f"Skill Extractor Agent\n\n*{p2}*\n\n✅ Skills Extracted")

            p3, analysis = analyze_roles(
                st.session_state.memory,
                st.session_state.vectorstore
            )
            st.markdown(f"Role Gap + Roadmap Agent\n\n*{p3}*\n\n✅ Analysis Ready")

            # ---------- FINAL REPORT ----------
            st.session_state.analysis = final_report(st.session_state.memory)
            st.session_state.ready = True


# ---------------- REPORT + CHAT ----------------
if st.session_state.ready:

    st.divider()
    st.markdown("## Career Report")
    st.markdown(st.session_state.analysis)

    st.divider()
    st.markdown("## 💬 Follow-Up Chat")

    # Display chat history
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a follow-up question...")

    if question:
        st.session_state.chat.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = chat_followup(
                    question,
                    st.session_state.memory,
                    st.session_state.vectorstore
                )
                st.markdown(answer)

        st.session_state.chat.append({"role": "assistant", "content": answer})
