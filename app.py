
import os
import uuid
import streamlit as st

from langgraph.types import Command
from agent import build_graph


# Page settings
st.set_page_config(
    page_title="Gift Finder AI",
    page_icon="🎁",
    layout="centered"
)

st.title("🎁 Gift Finder AI")
st.caption("Your personal AI gift recommendation assistant.")

# Load API key
if "GROQ_API_KEY" not in os.environ:
    st.error("GROQ_API_KEY is missing. Please configure your API key.")
    st.stop()

# Initialize agent
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_approval" not in st.session_state:
    st.session_state.pending_approval = False

graph = st.session_state.graph

config = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    }
}

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Approval interface
if st.session_state.pending_approval:

    st.warning("A deletion request is waiting for your approval.")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Confirm deletion"):
            result = graph.invoke(
                Command(resume=True),
                config
            )

            st.session_state.pending_approval = False
            st.rerun()

    with col2:
        if st.button("Cancel"):
            result = graph.invoke(
                Command(resume=False),
                config
            )

            st.session_state.pending_approval = False
            st.rerun()

# Chat input
if not st.session_state.pending_approval:

    prompt = st.chat_input("Who would you like to buy a gift for?")

    if prompt:

        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        result = graph.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            },
            config
        )

        if result.get("__interrupt__"):
            st.session_state.pending_approval = True
            st.rerun()

        answer = result["messages"][-1].content

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        st.rerun()
