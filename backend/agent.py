"""
LangChain Pandas Agent — Core agent module.

Creates a pandas DataFrame agent using Groq LLM that can:
- Answer natural language questions about the Titanic dataset
- Generate and execute Python/Pandas code deterministically
- Capture matplotlib visualizations as base64 images
"""

import io
import base64
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt

from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# AgentType import — handle different langchain versions
try:
    from langchain.agents.agent_types import AgentType
except ModuleNotFoundError:
    from langchain_classic.agents.agent_types import AgentType

# ─── System Prompt ────────────────────────────────────────────────
# This is injected before every LLM call so the model knows
# EXACTLY what columns exist, their types, and NaN gotchas.

SYSTEM_PREFIX = """You are a data analysis agent working with a pandas DataFrame called `df`.

The DataFrame contains Titanic passenger data with these EXACT columns:
- PassengerId (int): Unique passenger ID
- Survived (int): 0 = Did not survive, 1 = Survived
- Pclass (int): Ticket class — 1 = 1st, 2 = 2nd, 3 = 3rd
- Name (str): Full passenger name
- Sex (str): 'male' or 'female'
- Age (float): Age in years
- SibSp (int): Number of siblings/spouses aboard
- Parch (int): Number of parents/children aboard
- Ticket (str): Ticket number string
- Fare (float): Passenger fare paid
- Embarked (str): Port of embarkation — C = Cherbourg, Q = Queenstown, S = Southampton

RULES you MUST follow:
1. ALWAYS use the existing DataFrame `df` — never create new DataFrames from scratch.
2. Use ONLY the exact column names listed above.
3. ALWAYS execute Python code to compute the answer — never guess or make up numbers.
4. For visualizations: use `matplotlib.pyplot` directly — do NOT use seaborn.
   - Always do the full chart in ONE single code block.
   - Example for a bar chart:
     ```python
     import matplotlib.pyplot as plt
     data = df.groupby('Sex')['Survived'].mean()
     plt.figure(figsize=(8, 5))
     plt.bar(data.index, data.values, color=['steelblue', 'salmon'])
     plt.title('Survival Rate by Gender')
     plt.xlabel('Gender')
     plt.ylabel('Survival Rate')
     plt.tight_layout()
     ```
5. Complete the entire task in as few code executions as possible — ideally ONE.
6. When computing percentages, round to 2 decimal places.
7. After executing code and seeing the result, immediately provide the Final Answer.
"""


def create_agent(df):
    """
    Create a LangChain Pandas DataFrame agent.

    The agent uses Groq's LLM to interpret natural language questions,
    generate Python/Pandas code, execute it against the real DataFrame,
    and return computed results.

    Args:
        df: pandas DataFrame containing the Titanic dataset

    Returns:
        A LangChain agent executor ready to process queries
    """
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,  # Deterministic: same question → same code
    )

    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        prefix=SYSTEM_PREFIX,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        allow_dangerous_code=True,  # Required: agent executes generated Python
        verbose=True,               # Logs ReAct reasoning steps to console
        max_iterations=15,          # Allow enough steps for complex/chart queries
        max_execution_time=120,     # 2 minute hard timeout
        return_intermediate_steps=True, # Expose the generated code
        handle_parsing_errors=True, # Gracefully recover from malformed LLM output
    )

    return agent


def run_query(agent, question: str) -> dict:
    """
    Execute a user query through the agent and capture any plots.

    Pipeline:
    1. Clear previous matplotlib figures (prevent stale plot leakage)
    2. Run agent with user question (ReAct loop: Thought→Action→Observation)
    3. Check if any matplotlib figure was created during execution
    4. If yes: capture as PNG → base64 encode → include in response
    5. Return {"text": ..., "image": ... or None}

    Args:
        agent: The LangChain pandas agent executor
        question: User's natural language question

    Returns:
        dict with "text" (str) and "image" (str|None, base64-encoded PNG)
    """
    # 1. Clear any leftover figures from previous queries
    plt.close('all')

    try:
        # 2. Run the agent (this triggers the ReAct loop)
        result = agent.invoke({"input": question})
        answer_text = result.get("output", "I couldn't generate an answer.")
        
        # Extract the generated python code from intermediate steps
        code_blocks = []
        for action, _ in result.get("intermediate_steps", []):
            if action.tool == "python_repl_ast":
                code_blocks.append(action.tool_input)
        
        generated_code = "\n".join(code_blocks) if code_blocks else None
        
    except Exception as e:
        return {
            "text": f"I encountered an issue processing that query. Please try rephrasing. (Error: {str(e)})",
            "image": None,
            "code": None
        }

    # 3. Check if any matplotlib figure was created
    response = {"text": answer_text, "image": None, "code": generated_code}

    if plt.get_fignums():
        try:
            buf = io.BytesIO()
            fig = plt.gcf()
            fig.tight_layout()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            buf.seek(0)
            response["image"] = base64.b64encode(buf.read()).decode('utf-8')
        except Exception:
            pass  # If plot capture fails, just return text
        finally:
            plt.close('all')  # Release memory

    return response
