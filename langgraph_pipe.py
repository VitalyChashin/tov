from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
import requests


# ---------- 1. Shared state ----------
class GraphState(TypedDict):
    user_query: str
    weather_data: dict
    stock_data: dict
    messages: Annotated[Sequence[BaseMessage], add_messages]
    final_answer: str


# ---------- 2. Two deterministic API nodes (no LLM) ----------
def fetch_weather(state: GraphState) -> dict:
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": 55.75, "longitude": 37.62, "current": "temperature_2m"},
        timeout=10,
    )
    return {"weather_data": r.json()}


def fetch_stocks(state: GraphState) -> dict:
    r = requests.get("https://api.example.com/stocks/AAPL", timeout=10)
    return {"stock_data": r.json()}


# ---------- 3. Tools for the ReAct agent ----------
@tool
def search_news(query: str) -> str:
    """Search recent news for the given query."""
    return f"News results for: {query}"  # mock

@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression."""
    return str(eval(expression, {"__builtins__": {}}, {}))


# ---------- 4. Prebuilt ReAct agent (used as a sub-runnable) ----------
llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=0)
react_agent = create_react_agent(
    model=llm,
    tools=[search_news, calculator],
    prompt=(
        "You are an analyst. You will receive pre-fetched data plus a user query. "
        "Use the data and your tools to produce a grounded answer."
    ),
)


# ---------- 5. Node that hands the aggregated data to the agent ----------
def run_agent(state: GraphState) -> dict:
    context = (
        f"User query: {state['user_query']}\n\n"
        f"Pre-fetched weather: {state['weather_data']}\n"
        f"Pre-fetched stocks:  {state['stock_data']}\n"
    )
    result = react_agent.invoke({"messages": [HumanMessage(content=context)]})
    return {
        "messages": result["messages"],
        "final_answer": result["messages"][-1].content,
    }


# ---------- 6. Wire the graph ----------
builder = StateGraph(GraphState)
builder.add_node("fetch_weather", fetch_weather)
builder.add_node("fetch_stocks", fetch_stocks)
builder.add_node("run_agent", run_agent)

# Fan-out: both API calls run in the same super-step
builder.add_edge(START, "fetch_weather")
builder.add_edge(START, "fetch_stocks")

# Fan-in: run_agent waits until BOTH predecessors finish
builder.add_edge("fetch_weather", "run_agent")
builder.add_edge("fetch_stocks", "run_agent")

builder.add_edge("run_agent", END)

graph = builder.compile()


# ---------- 7. Run ----------
out = graph.invoke({"user_query": "Should I buy AAPL today given the weather in Moscow?"})
print(out["final_answer"])