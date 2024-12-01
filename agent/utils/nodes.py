from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.utilities import PythonREPL
from .state import AgentState
import json
from typing import Literal
from .tools import complete_python_task, think_or_plan
from langgraph.prebuilt import ToolInvocation, ToolExecutor
import base64
import copy


llm = ChatOpenAI(model="gpt-4o")

tools = [complete_python_task, think_or_plan]

model = llm.bind_tools(tools)

tool_executor = ToolExecutor(tools)

prompt = """# Role
You are a creative problem solver and a highly experienced expert python developer.

You are completing some programming puzzles that are christmas themed and designed to test your programming and creative problem solving abilities.

# Input

You will be given the following input:
- The full problem description
- The problem input file path
- A sample of the problem input file
- The problem output file path

The problem description will be detailed and contain a simplier input and output example that you can use to test your code.

# Output

You should use the `complete_python_task` tool to submit your final solution, which means saving your solution to the problem output file path. There should be seperate solutions for part one and part two, formatted like:

```
Part One: <solution>
Part Two: <solution>
```

# Approach

The problems are really difficult so you should deconstruct the problem into its fundamental parts and identify the best algorithm for solving the problem.

You should always extract the example input and output from the problem description and use it to test your code. You should then generally follow this approach:
1. Think about the problem and plan your approach
2. Write the code to solve the problem
3. Test the code with the example input and output
4. Rinse and repeat until the problem is solved

# Tone

All of your thoughts and explanations should be in MARKDOWN format, and they should be christmas themed and festive, in the same style as the problem description and continue the narrative of the problem description.

# Capabilities
1. **Execute python code** using the `complete_python_task` tool. Using valid Markdown for thoughts and explanations.
2. **Think about the next action or plan your approach** using the `think_or_plan` tool.

# Python Code Guidelines
- **ALL INPUT DATA IS LOADED ALREADY**, so use the provided variable names to access the data.
- **VARIABLES PERSIST BETWEEN RUNS**, so reuse previously defined variables if needed.
- **TO SEE CODE OUTPUT**, use `print()` statements. You won't be able to see outputs of `pd.head()`, `pd.describe()` etc. otherwise.
- **ONLY USE THE FOLLOWING LIBRARIES**:
  - `pandas`
"""
# 3. **Create SVG plots** using the `create_svg` tool and pass the SVG code to it. Using valid Markdown for thoughts and explanations.

# ## SVGs
# - Use the thought field to talk about the SVG you are creating and how you are going to create it.
# - You can create SVGs using the `create_svg` tool.
# - They should be valid SVG code, so no extra text or formatting.
# SVGs are more suitable for displaying data in a more visual way, such as charts and diagrams.
chat_template = ChatPromptTemplate.from_messages([
    ("system", prompt),
    ("placeholder", "{messages}"),
])
model = chat_template | model



def route_to_tools(
    state: AgentState,
) -> Literal["tools", "__end__"]:
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route back to the agent.
    """

    if messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "__end__"


def call_model(state: AgentState):
    print("Calling model")
    print(state)
    state["messages"] = state["messages"]
    llm_outputs = model.invoke(state)
    print(llm_outputs)
    return {"messages": [llm_outputs]}


def call_tools(state: AgentState):
    last_message = state["messages"][-1]
    tool_invocations = []
    if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls'):
        tool_invocations = [
            ToolInvocation(
                tool=tool_call["name"],
                tool_input={**tool_call["args"], "graph_state": state}
            ) for tool_call in last_message.tool_calls
        ]

    responses = tool_executor.batch(tool_invocations, return_exceptions=True)
    tool_messages = []
    state_updates = {}

    for tc, response in zip(last_message.tool_calls, responses):
        if isinstance(response, Exception):
            raise response
        message, updates = response
        tool_messages.append(ToolMessage(
            content=str(message),
            name=tc["name"],
            tool_call_id=tc["id"]
        ))
        state_updates.update(updates)

    if 'messages' not in state_updates:
        state_updates["messages"] = []

    state_updates["messages"] = tool_messages #+ state_updates["messages"]
    return state_updates

