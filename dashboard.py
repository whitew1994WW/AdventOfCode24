import streamlit as st
import os

with open('.env') as f:
    for line in f:
        key, value = line.strip().split('=', 1)
        os.environ[key] = value

from agent.agent import AdventOfCodeAgent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid


# Set up session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Main chat interface
chat_container = st.container(height=680)

def display_message(msg, intermediate_outputs=None, tool_index=0):
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)
    elif isinstance(msg, ToolMessage):
        if intermediate_outputs and len(intermediate_outputs) > tool_index:
            output = intermediate_outputs[tool_index]
            if output["type"] == "python_output":
                with st.expander("Executing Python Code..."):
                    st.markdown("**Thought:**")
                    st.markdown(output["thought"])
                    st.markdown("**Code:**")
                    st.code(output["code"])
                    st.markdown("**Output:**")
                    st.code(output["output"])
            elif output["type"] == "python_error":
                with st.expander("Python Error"):
                    st.markdown("**Thought:**")
                    st.markdown(output["thought"])
                    st.markdown("**Code:**")
                    st.code(output["code"]) 
                    st.markdown("**Error:**")
                    st.error(output["output"])
            elif output["type"] == "thought":
                with st.expander("Thinking and Planning..."):
                    st.markdown(output["thought"])

def display_chat():
    with chat_container:
        tool_count = 0
        for msg in st.session_state.chat_history:
            if isinstance(msg, ToolMessage) and st.session_state.agent:
                display_message(msg, st.session_state.agent.intermediate_outputs, tool_count)
                tool_count += 1
            else:
                display_message(msg)

def handle_chat_input(user_input):
    if st.session_state.agent:
        chat_placeholder = st.empty()
        with chat_container:
            tool_count = 0
            for msg, _ in st.session_state.agent.user_sent_message(user_input):
                st.session_state.chat_history = st.session_state.agent.chat_history
                if isinstance(msg, ToolMessage):
                    display_message(msg, st.session_state.agent.intermediate_outputs, tool_count)
                    tool_count += 1
                else:
                    display_message(msg)
        st.rerun()
    else:
        st.warning("Please select a day and start the conversation first!")


# Sidebar
with st.sidebar:
    st.title("Advent of Code Assistant")
    
    # Get list of input files
    input_files = os.listdir(os.path.join(os.path.dirname(__file__), "problem_dataset", "problem_inputs"))
    days = [int(f.replace("day_", "").replace(".txt", "")) for f in input_files]
    days.sort()
    
    selected_day = st.selectbox("Select Day", days)
    
    if st.button("Start Conversation"):
        st.session_state.agent = AdventOfCodeAgent(selected_day)
        with chat_container:
            tool_count = 0
            for msg, _ in st.session_state.agent.start_conversation():
                st.session_state.chat_history = st.session_state.agent.chat_history
                if isinstance(msg, ToolMessage):
                    display_message(msg, st.session_state.agent.intermediate_outputs, tool_count)
                    tool_count += 1
                else:
                    display_message(msg)
        st.rerun()


# Display chat history
display_chat()

# Chat input
if "chat_input_key" not in st.session_state:
    st.session_state.chat_input_key = str(uuid.uuid4())
    
chat_input = st.chat_input("Enter your message here", key=st.session_state.chat_input_key)
if chat_input:
    handle_chat_input(chat_input)
