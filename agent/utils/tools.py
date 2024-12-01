from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
import pandas as pd
from langchain_core.messages import AIMessage
from typing import Annotated, Tuple
from langgraph.prebuilt import InjectedState
import sys
from io import StringIO

repl = PythonREPL()

persistent_vars = {}

@tool(parse_docstring=True)
def complete_python_task(
        graph_state: Annotated[dict, InjectedState], thought: str, python_code: str
) -> Tuple[str, dict]:
    """Completes a python task

    Args:
        thought: Internal thought about the next action to be taken, and the reasoning behind it. This should be formatted in MARKDOWN and be high quality.
        python_code: Python code to be executed
    """
    current_variables = graph_state["current_variables"] if "current_variables" in graph_state else {}
    
    try:
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        # Execute the code and capture the result
        exec_globals = globals().copy()
        exec_globals.update(persistent_vars)
        exec_globals.update(current_variables)

        exec(python_code, exec_globals)
        persistent_vars.update({k: v for k, v in exec_globals.items() if k not in globals()})

        # Get the captured stdout
        output = sys.stdout.getvalue()

        # Restore stdout
        sys.stdout = old_stdout

        updated_state = {
            "intermediate_outputs": [{"type": "python_output", "thought": thought, "code": python_code, "output": output}],
            "current_variables": persistent_vars
        }

        return output, updated_state
    except Exception as e:
        return str(e), {"intermediate_outputs": [{"type": "python_error", "thought": thought, "code": python_code, "output": str(e)}]}
    
@tool(parse_docstring=True)
def think_or_plan(
    graph_state: Annotated[dict, InjectedState], thought: str
) -> Tuple[str, dict]:
    """Think about the next action or plan your approach to solving the problem
    
    Args:
        thought: Internal thought about the next action to be taken, and the reasoning behind it. This should be formatted in MARKDOWN and be high quality.
    """
    return thought, {"intermediate_outputs": [{"type": "thought", "thought": thought}]}
