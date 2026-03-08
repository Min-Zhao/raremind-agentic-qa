"""
chatbot_app.py – Streamlit chatbot UI for the Agentic Rare Disease QA Pipeline.

Launch:
    streamlit run app/chatbot_app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

# Make sure src/ is importable when running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.planning_agent import AgentResponse, PlanningAgent
from src.utils.config_loader import load_config

# ────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RareMind – Rare Disease AI Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .main { background-color: #f8fafc; }
    .stChatMessage { border-radius: 12px; padding: 4px 0; }
    .route-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px;
    }
    .route-rag    { background: #dbeafe; color: #1e40af; }
    .route-web    { background: #dcfce7; color: #166534; }
    .route-mcp    { background: #fef9c3; color: #92400e; }
    .route-history{ background: #ede9fe; color: #4c1d95; }
    .route-hybrid { background: #fce7f3; color: #9d174d; }
    .trace-box {
        background: #1e293b;
        color: #94a3b8;
        border-radius: 8px;
        padding: 12px;
        font-family: monospace;
        font-size: 0.8rem;
        margin-top: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading RareMind pipeline…")
def get_agent() -> PlanningAgent:
    config = load_config()
    return PlanningAgent(config=config)


def init_session() -> None:
    defaults = {
        "messages": [],       # list of {role, content, meta}
        "show_trace": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session()

# ────────────────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/color/96/dna-helix.png", width=60)
    st.title("RareMind")
    st.caption("Rare Disease AI Assistant")
    st.divider()

    st.markdown("### Settings")
    show_trace = st.toggle("Show reasoning trace", value=False)
    st.session_state["show_trace"] = show_trace

    st.divider()
    if st.button("🗑️  Clear conversation", use_container_width=True):
        try:
            get_agent().reset_memory()
        except Exception:
            pass
        st.session_state["messages"] = []
        st.rerun()

    st.divider()
    st.markdown(
        """
        **Supported topics**
        - Complex Lymphatic Anomalies (CLA)
        - Gorham-Stout Disease
        - Generalized Lymphatic Anomaly
        - Kaposiform Lymphangiomatosis
        - Clinical trials & treatments
        - Patient support resources
        """
    )

    st.divider()
    st.caption(
        "⚠️ This tool provides general information only and does not constitute "
        "medical advice. Always consult a qualified healthcare professional."
    )

# ────────────────────────────────────────────────────────────────────────────
# Main chat area
# ────────────────────────────────────────────────────────────────────────────

st.title("🩺 RareMind – Rare Disease AI Assistant")
st.caption(
    "Ask me anything about rare diseases, treatments, clinical trials, and support resources."
)

# Render chat history
for msg in st.session_state["messages"]:
    avatar = "🧑" if msg["role"] == "user" else "🩺"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        meta = msg.get("meta", {})
        if meta and msg["role"] == "assistant":
            route = meta.get("route", "")
            confidence = meta.get("confidence", 0.0)
            latency = meta.get("latency_ms", 0.0)
            sources = meta.get("sources", [])

            route_class = f"route-{route}" if route in (
                "rag", "web", "mcp", "history", "hybrid"
            ) else "route-rag"
            st.markdown(
                f'<span class="route-badge {route_class}">🔀 {route.upper()}</span>'
                f'<span class="route-badge" style="background:#f1f5f9;color:#475569">'
                f'⏱ {latency:.0f}ms</span>'
                f'<span class="route-badge" style="background:#f1f5f9;color:#475569">'
                f'🎯 {confidence:.0%}</span>',
                unsafe_allow_html=True,
            )

            if sources:
                with st.expander("📚 Sources", expanded=False):
                    for src in sources[:5]:
                        label = src.get("label", "")
                        src_type = src.get("type", "")
                        icon = {"rag": "📄", "web": "🌐", "mcp": "🔬", "history": "💬"}.get(src_type, "📌")
                        st.markdown(f"{icon} {label}")

            if show_trace and meta.get("trace"):
                with st.expander("🔍 Reasoning trace", expanded=False):
                    trace_lines = []
                    for step in meta["trace"]:
                        trace_lines.append(
                            f"Step {step['step']} [{step['agent']}] {step['action']}\n"
                            f"  → {step['result_summary']} ({step['duration_ms']:.0f}ms)"
                        )
                    st.markdown(
                        f'<div class="trace-box">' +
                        "<br>".join(trace_lines) +
                        "</div>",
                        unsafe_allow_html=True,
                    )

# Chat input
if query := st.chat_input("Ask about a rare disease…"):
    # Append user message
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(query)

    # Run pipeline
    with st.chat_message("assistant", avatar="🩺"):
        with st.spinner("RareMind is thinking…"):
            try:
                agent = get_agent()
                t0 = time.perf_counter()
                response: AgentResponse = agent.run(query)
                latency = (time.perf_counter() - t0) * 1000

                st.markdown(response.final_answer)

                # Route / confidence badges
                route = response.route
                route_class = f"route-{route}" if route in (
                    "rag", "web", "mcp", "history", "hybrid"
                ) else "route-rag"
                st.markdown(
                    f'<span class="route-badge {route_class}">🔀 {route.upper()}</span>'
                    f'<span class="route-badge" style="background:#f1f5f9;color:#475569">'
                    f'⏱ {latency:.0f}ms</span>'
                    f'<span class="route-badge" style="background:#f1f5f9;color:#475569">'
                    f'🎯 {response.confidence:.0%}</span>',
                    unsafe_allow_html=True,
                )

                # Sources expander
                if response.sources:
                    with st.expander("📚 Sources", expanded=False):
                        for src in response.sources[:5]:
                            label = src.get("label", "")
                            src_type = src.get("type", "")
                            icon = {"rag": "📄", "web": "🌐", "mcp": "🔬", "history": "💬"}.get(
                                src_type, "📌"
                            )
                            st.markdown(f"{icon} {label}")

                # Trace expander
                if show_trace and response.trace:
                    with st.expander("🔍 Reasoning trace", expanded=False):
                        trace_lines = []
                        for step in response.trace:
                            trace_lines.append(
                                f"Step {step.step} [{step.agent}] {step.action}\n"
                                f"  → {step.result_summary} ({step.duration_ms:.0f}ms)"
                            )
                        st.markdown(
                            '<div class="trace-box">' +
                            "<br>".join(trace_lines) +
                            "</div>",
                            unsafe_allow_html=True,
                        )

                # Medical emergency banner
                if response.is_medical_emergency:
                    st.error(
                        "⚠️ This appears to be a medical emergency. "
                        "Please call 911 or your local emergency number immediately."
                    )

                # Persist to session state
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": response.final_answer,
                    "meta": {
                        "route": response.route,
                        "confidence": response.confidence,
                        "latency_ms": latency,
                        "sources": response.sources,
                        "trace": [
                            {
                                "step": s.step,
                                "agent": s.agent,
                                "action": s.action,
                                "result_summary": s.result_summary,
                                "duration_ms": s.duration_ms,
                            }
                            for s in response.trace
                        ],
                    },
                })

            except Exception as exc:
                err_msg = (
                    f"An error occurred: {exc}\n\n"
                    "Please check your API keys and configuration."
                )
                st.error(err_msg)
                st.session_state["messages"].append(
                    {"role": "assistant", "content": err_msg}
                )
