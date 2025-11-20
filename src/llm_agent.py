import urllib.parse
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_community.utilities import SQLDatabase
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI
import streamlit as st

from .config import LLM_MODEL_NAME, OPENAI_API_KEY, DATABASE

CUSTOM_SUFFIX = """Begin!

Relevant pieces of previous conversation:
{chat_history}
(Note: Only reference this information if it is relevant to the current query.)

Question: {input}
Thought Process: It is imperative that I do not fabricate information not present in any table or engage in hallucination; maintaining trustworthiness is crucial.
In SQL queries involving string or TEXT comparisons like first_name, I must use the `LOWER()` function for case-insensitive comparisons and the `LIKE` operator for fuzzy matching. 
Queries for return percentage is defined as total number of returns divided by total number of orders. You can join orders table with users table to know more about each user.
Make sure that query is related to the SQL database and tables you are working with.
If the result is empty, the Answer should be "No results found". DO NOT hallucinate an answer if there is no result.

CRITICAL: When you execute a query that returns data for visualization, your Final Answer MUST include the raw data results.

For visualization queries, format your Final Answer as:
"[Description of what was found]

Data: [('item1', value1), ('item2', value2), ...]"

DO NOT just describe the data - include the complete actual query results in your Final Answer.

{agent_scratchpad}
"""

FORMAT_INSTRUCTIONS = """Use the following format:

Input: the input question you must answer
Thought: <Reasoning for what the next step should be>
Action: the action to take, should be one of [{tool_names}] if using a tool, otherwise answer on your own.
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Final Thought: <Final reasoning to collate the answer for Input>
Final Answer: <the final answer to the original input question>"""

langchain_chat_kwargs = {
    "temperature": 0,
    "max_tokens": 4000,
    "verbose": True,
}
chat_openai_model_kwargs = {
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": -1,
}

class ValidatingPythonREPLTool(PythonREPLTool):
    """Runs Python code to validate it, but hides execution output."""

    def _run(self, code: str, **kwargs):
        try:
            super()._run(code, **kwargs)  # actually executes
            return "✅ Code executed without errors."
        except Exception as e:
            return f"❌ Validation failed: {e}"

def get_chat_openai(model_name):
    """
    Returns an instance of the ChatOpenAI class initialized with the specified model name.

    Args:
        model_name (str): The name of the model to use.

    Returns:
        ChatOpenAI: An instance of the ChatOpenAI class.

    """
    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        max_tokens=4000
    )
    return llm


def get_sql_toolkit(tool_llm_name: str):
    """
    Instantiates a SQLDatabaseToolkit object with the specified language model.

    This function creates a SQLDatabaseToolkit object configured with a language model
    obtained by the provided model name. The SQLDatabaseToolkit facilitates SQL query
    generation and interaction with a database.

    Args:
        tool_llm_name (str): The name or identifier of the language model to be used.

    Returns:
        SQLDatabaseToolkit: An instance of SQLDatabaseToolkit initialized with the provided language model.
    """
    # Use the shared database connection from session state if available
    if hasattr(st, 'session_state') and 'db_connection' in st.session_state:
        try:
            conn = st.session_state['db_connection']
            db = SQLDatabase(conn.session.bind)
        except Exception:
            db = SQLDatabase.from_uri(f"sqlite:///{DATABASE}")
    else:
        # Try Streamlit connection or fallback
        try:
            conn = st.connection("ecommerce_db", type="sql", url=f"sqlite:///{DATABASE}")
            db = SQLDatabase(conn.session.bind)
        except Exception:
            db = SQLDatabase.from_uri(f"sqlite:///{DATABASE}")

    llm_tool = get_chat_openai(model_name=tool_llm_name)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm_tool)
    return toolkit


def get_agent_llm(agent_llm_name: str):
    """
    Retrieve a language model agent for conversational tasks.

    Args:
        agent_llm_name (str): The name or identifier of the language model for the agent.

    Returns:
        ChatOpenAI: A language model agent configured for conversational tasks.
    """
    llm_agent = get_chat_openai(model_name=agent_llm_name)
    return llm_agent


def initialize_python_agent(agent_llm_name: str = LLM_MODEL_NAME):
    """
    Create an agent for Python-related tasks.

    Args:
        agent_llm_name (str): The name or identifier of the language model for the agent.

    Returns:
        AgentExecutor: An agent executor configured for Python-related tasks.

    """
    instructions = """You are an agent designed to write Python code to create visualizations from SQL query results.
            You have access to a python REPL, which you can use to execute python code.
            
            IMPORTANT: You will receive actual SQL query results as input. DO NOT create sample data.
            Use the EXACT data provided from the SQL query results.
            
            Your task:
            1. Parse the actual SQL query results provided in the input
            2. Convert the data into the appropriate format for Plotly
            3. Create a visualization using only Plotly (no matplotlib)
            4. Return ONLY the Python code in the format ```python <code>```
            
            The SQL results will be in format like: [('State1', 100), ('State2', 200), ...]
            You must use this actual data, not create sample data.
            """
    tools = [ValidatingPythonREPLTool()]
    base_prompt = hub.pull("langchain-ai/openai-functions-template")
    prompt = base_prompt.partial(instructions=instructions)
    agent = create_openai_functions_agent(ChatOpenAI(model=agent_llm_name, temperature=0), tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    return agent_executor


def initialize_sql_agent(tool_llm_name: str = LLM_MODEL_NAME, agent_llm_name: str = LLM_MODEL_NAME):
    """
    Create an agent for SQL-related tasks.

    Args:
        tool_llm_name (str): The name or identifier of the language model for SQL toolkit.
        agent_llm_name (str): The name or identifier of the language model for the agent.

    Returns:
        Agent: An agent configured for SQL-related tasks.

    """
    llm_agent = get_agent_llm(agent_llm_name)
    toolkit = get_sql_toolkit(tool_llm_name)
    message_history = SQLChatMessageHistory(
        session_id="my-session",
        connection_string=f"sqlite:///session_history.db",
        table_name="message_store",
        session_id_field_name="session_id"
    )
    memory = ConversationBufferMemory(memory_key="chat_history", input_key='input', chat_memory=message_history, return_messages=False)

    agent = create_sql_agent(
        llm=llm_agent,
        toolkit=toolkit,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        input_variables=["input", "agent_scratchpad", "chat_history"],
        suffix=CUSTOM_SUFFIX,
        format_instructions = FORMAT_INSTRUCTIONS,
        memory=memory,
        agent_executor_kwargs={"memory": memory, "handle_parsing_errors": True},
        verbose=True
    )
    return agent
