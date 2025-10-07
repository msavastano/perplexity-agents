# Import necessary standard libraries
import asyncio  # For running asynchronous code
import os       # To access environment variables
from dotenv import load_dotenv
import logging

# Import AsyncOpenAI for creating an async client
from openai import AsyncOpenAI

# Import custom classes and functions from the agents package.
from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load local .env file (if present) then retrieve configuration from environment variables
load_dotenv()

# Retrieve configuration from environment variables or use defaults
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME") or "gpt-4"

# Validate that all required configuration variables are set
if not API_KEY:
    raise ValueError("Please set OPENAI_API_KEY.")

# Initialize the custom OpenAI async client with the specified API_KEY.
client = AsyncOpenAI(api_key=API_KEY)

# Disable tracing to avoid using a platform tracing key; adjust as needed.
set_tracing_disabled(disabled=True)

# Define the specialized agents

# Web Searcher Agent
web_searcher_agent = Agent(
    name="WebSearcher",
    description="Use this agent for questions that require up-to-date information or access to the internet.",
    instructions="You are a web searcher. Use the web_search tool to answer the user's question.",
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    tools=[{"type": "web_search"}],
)

# Basic Agent
basic_agent = Agent(
    name="BasicAgent",
    description="Use this agent for general questions, conversation, or tasks that do not require web access.",
    instructions="You are a helpful assistant. Answer the user's question directly.",
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
)

# Orchestrator Agent
# This agent's purpose is to decide which specialized agent to hand off to.
# The `handoffs` parameter makes the other agents available as tools for delegation.
orchestrator_agent = Agent(
    name="Orchestrator",
    instructions=(
        "You are an orchestrator. Your job is to analyze the user's query and decide which agent is best suited to answer it. "
        "Based on the query, you must call the appropriate agent to handle the request."
    ),
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    handoffs=[web_searcher_agent, basic_agent],
)

async def main():
    """
    Main asynchronous function to set up and run the agent.
    """
    query = input("Please enter your query: ")

    logger.info(f"Running orchestrator with model={MODEL_NAME!r}")

    # Run the orchestrator, which will automatically hand off to the appropriate agent
    result = await Runner.run(orchestrator_agent, query)

    # Print the final output from the agent.
    logger.info("Orchestration complete.")
    print(result.final_output)

# Standard boilerplate to run the async main() function.
if __name__ == "__main__":
    # Import nest_asyncio to support nested event loops
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(main())