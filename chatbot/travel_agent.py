from pathlib import Path

import chromadb
from agents import (
    Agent,
    FunctionTool,
    function_tool,
)

import dotenv
dotenv.load_dotenv()

MODEL = "litellm/bedrock/eu.amazon.nova-lite-v1:0"


def bedrock_tool(tool: dict) -> FunctionTool:
    """Converts an OpenAI Agents SDK function_tool to a Bedrock-compatible FunctionTool."""
    return FunctionTool(
        name=tool["name"],
        description=tool["description"],
        params_json_schema={
            "type": "object",
            "properties": {
                k: v for k, v in tool["params_json_schema"]["properties"].items()
            },
            "required": tool["params_json_schema"].get("required", []),
        },
        on_invoke_tool=tool["on_invoke_tool"],
    )


chroma_path = Path(__file__).parent.parent / "chroma"
chroma_client = chromadb.PersistentClient(path=str(chroma_path))
city_db = chroma_client.get_collection(name="city_db")


@function_tool
def city_lookup_tool(query: str, max_results: int = 3) -> str:
    """
    Tool function for a RAG database to look up city information for specific cities.

    Args:
        query: The city information to look up.
        max_results: The maximum number of results to return.

    Returns:
        A string containing the city information.
    """

    results = city_db.query(query_texts=[query], n_results=max_results)

    if not results["documents"][0]:
        return f"No travel information found for: {query}"

    return "City Information:\n" + "\n".join(results)


travel_agent = Agent(
    name="Travel Assistant",
    instructions="""
    You are a helpful travel assistant giving out city information.
    You give concise answers.
    If you need to look up city information, use the city_lookup_tool.
    """,
    model=MODEL,
    tools=[bedrock_tool(city_lookup_tool.__dict__)],
)
