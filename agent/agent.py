import json
import os
from langgraph.graph import StateGraph

from .utils.state import AgentState
from .utils.nodes import call_model, call_tools, route_to_tools
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime

problem_description_folder = os.path.join(os.path.dirname(__file__), "..", "problem_dataset", "problem_descriptions")
problem_input_folder = os.path.join(os.path.dirname(__file__), "..", "problem_dataset", "problem_inputs") 
problem_output_folder = os.path.join(os.path.dirname(__file__), "..", "problem_dataset", "problem_outputs")

class AdventOfCodeAgent():
    def __init__(self, problem_day: int):
        super().__init__()
        self.chat_history = []
        self.intermediate_outputs = []
        self.graph = self.create_graph()
        self.graph_state = {}
        self.problem_day = problem_day
        self.problem_input = self.get_problem_input()
        self.problem_description = self.get_problem_description()
        self.problem_output_file_path = self.get_problem_output_file_path()

    def get_problem_input(self):
        self.problem_input_file_path = os.path.join(problem_input_folder, f"day_{self.problem_day}.txt").replace('/', '\\\\')
        with open(self.problem_input_file_path, "r") as file:
            self.problem_input_lines = file.readlines()
            return file.read()

    def get_problem_description(self):
        self.problem_description_file_path = os.path.join(problem_description_folder, f"day_{self.problem_day}.txt").replace('/', '\\\\')
        with open(self.problem_description_file_path, "r") as file:
            return file.read()
    
    def get_problem_output_file_path(self):
        self.problem_output_file_path = os.path.join(problem_output_folder, f"day_{self.problem_day}.txt").replace('/', '\\\\')
        return self.problem_output_file_path

    def create_graph(self):
        # Delete the images folder
        if not os.path.exists("images"):
            os.makedirs("images")
        workflow = StateGraph(AgentState)
        workflow.add_node('agent', call_model)
        workflow.add_node('tools', call_tools)

        workflow.add_conditional_edges('agent', route_to_tools)

        workflow.add_edge('tools', 'agent')
        workflow.set_entry_point('agent')
        return workflow.compile()
    
    def user_sent_message(self, user_query):
        input_state = {
            "messages": self.chat_history + [HumanMessage(content=user_query)], 
            "current_date": datetime.now().strftime("%Y-%m-%d"),
            "intermediate_outputs": self.intermediate_outputs,
            "problem_input": self.problem_input,
            "problem_description": self.problem_description
        }
        for result in self.graph.stream(input=input_state, config={"recursion_limit": 15}, stream_mode='values'):
            self.chat_history = result["messages"]
            self.graph_state = result
            self.intermediate_outputs = result["intermediate_outputs"]
            
            yield result['messages'][-1], len(self.chat_history) - 1


    def start_conversation(self):
        problem_input_lines = ''.join(self.problem_input_lines[:10])
        user_query = f"The problem description is as follows:\n```\n{self.problem_description}\n```\n\nThe problem input path is: {self.problem_input_file_path}\n\nThe problem output path is: {self.problem_output_file_path}\n\nThe first 10 lines of the problem input are:\n```\n{problem_input_lines}\n...\n```"
        input_state = {
            "messages": self.chat_history + [HumanMessage(content=user_query)], 
            "current_date": datetime.now().strftime("%Y-%m-%d"),
            "intermediate_outputs": self.intermediate_outputs,
            "problem_input": self.problem_input,
            "problem_description": self.problem_description
        }
        for result in self.graph.stream(input=input_state, config={"recursion_limit": 15}, stream_mode='values'):
            self.chat_history = result["messages"]
            self.graph_state = result
            self.intermediate_outputs = result["intermediate_outputs"]
            
            yield result['messages'][-1], len(self.chat_history) - 1




