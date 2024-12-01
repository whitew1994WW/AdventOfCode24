import operator
from typing import Sequence, TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
from typing import Dict, Optional

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_variables: dict
    intermediate_outputs: Annotated[List[Dict[str, str]], operator.add]
    problem_input: str
    problem_description: str
    problem_output_file: str

