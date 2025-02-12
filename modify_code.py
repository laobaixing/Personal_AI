import json
import os
import re
import subprocess

from langchain_core.messages import HumanMessage
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel


# Define a state that holds conversation messages and file/code data.
class GraphState(BaseModel):
    messages: list = add_messages([], [])  # Conversation messages
    plan_steps: list = []  # e.g., ["read_file", "modify_code"]
    file_path: str = ""  # File path extracted from the prompt
    file_content: str = ""  # Original code read from the file
    modified_code: str = ""  # Code generated by the LLM
    save_status: str = ""  # Status message after saving the code
    execution_result: str = ""  # Output of running the saved code


# Import secrets for Vertex AI and OpenAI
secret = json.load(open("secret.json"))
PROJECT_ID = secret["project_id"]
LOCATION = secret["location"]
MODEL_NAME = "gemini-2.0-pro-exp-02-05"

# Initialize the LLM (this will pick up the OPENAI_API_KEY from the environment)
llm = ChatVertexAI(
    project_id=PROJECT_ID, location=LOCATION, model=MODEL_NAME, temperature=0
)


# Node 0: GetPromptNode
# Reads a prompt from the terminal and appends it as a HumanMessage.
def get_prompt_node(state: GraphState) -> dict:
    user_input = input("Please enter your prompt: ")
    return {"messages": [HumanMessage(content=user_input)]}


# Node 1: PlanStepsNode
# Uses the LLM to break down the instruction and extract actionable steps and a file path.
def plan_steps_node(state: GraphState) -> dict:
    print("-----plan steps------\n")
    prompt = state.messages[-1].content if state.messages else ""
    plan_prompt = f"""
        Analyze the following instruction and break it down into actionable steps.
        If the instruction involves modifying a file (e.g. 'adding features to a file'),
        Output a Json object with keys 'steps' including two member 'read_file' and 'modify_code'.
        Also, if a file path is mentioned, extract it and put it in the above Json object with key 'file_path'.
        Please note the file name itself could be a file path.
        The Json answer could be this format: {{'steps': ['read_file', 'modify_code'], 'file_path': 'file_path'}}\n
        Instruction: {prompt}"""
    response = llm.invoke([HumanMessage(content=plan_prompt)]).content.strip()

    # Extract JSON using regex
    match = re.search(r"\{.*\}", response, re.DOTALL)  # Match anything between { and }
    if match:
        try:
            json_str = match.group(0)
            result = json.loads(json_str)
            steps = result.get("steps", [])
            file_path = result.get("file_path", "")
            file_path = os.path.expanduser(file_path)
        except Exception as e:
            steps = []
            file_path = ""
            print(f"Error: {e}")
    else:
        print("No valid JSON found")

    return {"plan_steps": steps, "file_path": file_path}


# Node 2: ReadFileNode
# Reads the file at the extracted file path to retrieve the original code.
def read_file_node(state: GraphState) -> dict:

    file_path = state.file_path
    print(f"-----read the file: {file_path}------\n")

    if not file_path:
        return {"file_content": "No file path provided by LLM."}
    try:
        with open(file_path, "r") as f:
            content = f.read()
    except Exception as e:
        content = f"Error reading file: {e}"
    return {"file_content": content}


# Node 3: ModifyCodeNode
# Uses the LLM to generate updated code based on the original instruction and file content.
def modify_code_node(state: GraphState) -> dict:
    original_prompt = state.messages[-1].content if state.messages else ""
    file_info = state.file_content
    mod_prompt = (
        "You are a code assistant. Given the following instruction and code, generate updated Python code. "
        "Include comments that explain how the instruction influenced the changes.\n"
        f"Prompt: {original_prompt}\n"
        f"Code Content: {file_info}\n"
        "Output the complete new code. The output code should be able to run through command line"
    )
    print("-----modify the code-----\n")
    modified = llm.invoke([HumanMessage(content=mod_prompt)]).content

    if state.file_path[-2:] == "py":
        match = re.search(r"```python\s*\n(.*?)```", modified, re.DOTALL)
    if state.file_path[-3:] == "sql":
        match = re.search(r"```sql\s*\n(.*?)```", modified, re.DOTALL)

    if match:
        new_code = match.group(1)  # Extract the code part
    else:
        print("No Python code found.")
    return {"modified_code": new_code}


# Node 4: SaveCodeNode
# Saves the modified code back to the file.
def save_code_node(state: GraphState) -> dict:
    file_path = state.file_path
    print("-----save the code-----\n")
    if not file_path:
        # Optionally, set a default file name if none was extracted.
        file_path = "modified_code.py"
        state.file_path = file_path
    try:
        with open(file_path, "w") as f:
            f.write(state.modified_code)
        result = f"Code saved to {file_path}."
    except Exception as e:
        result = f"Error saving code: {e}"
    return {"save_status": result}


# Node 5: RunCodeCommandNode
# Executes the saved file via the command line.
def run_code_command_node(state: GraphState) -> dict:
    file_path = state.file_path
    print("------run the code-------\n")
    if file_path[-2:] == "py":
        try:
            # Run the file as a subprocess and capture its output.
            result = subprocess.run(
                ["python", file_path], capture_output=True, text=True, check=True
            )
            output = result.stdout
        except Exception as e:
            output = f"Error running file: {e}"
    else:
        output = "This is not a python file."
    return {"execution_result": output}


# Build the workflow graph.
graph = StateGraph(GraphState)

# Add nodes with names appended by "_node" to avoid conflicts with state keys.
graph.add_node("get_prompt_node", get_prompt_node)
graph.add_node("plan_steps_node", plan_steps_node)
graph.add_node("read_file_node", read_file_node)
graph.add_node("modify_code_node", modify_code_node)
graph.add_node("save_code_node", save_code_node)
graph.add_node("run_code_command_node", run_code_command_node)

# Set up the graph connections:
graph.add_edge(START, "get_prompt_node")
graph.add_edge("get_prompt_node", "plan_steps_node")


# Conditional routing: if "read_file" is among the planned steps, go to read_file_node; else, proceed directly to modify_code_node.
def steps_router(state: GraphState):
    print(state)
    if "read_file" in state.plan_steps:
        return "read_file_node"
    else:
        return "modify_code_node"


graph.add_conditional_edges(
    "plan_steps_node",
    steps_router,
    {"read_file_node": "read_file_node", "modify_code_node": "modify_code_node"},
)

graph.add_edge("read_file_node", "modify_code_node")
graph.add_edge("modify_code_node", "save_code_node")
graph.add_edge("save_code_node", "run_code_command_node")
graph.add_edge("run_code_command_node", END)

if __name__ == "__main__":
    initial_state = GraphState()
    # Compile the graph into a runnable object.
    compiled_graph = graph.compile()
    state = compiled_graph.invoke(initial_state)
    print("Final Execution Result:", state.get("execution_result"))
