# Import necessary standard libraries
import asyncio  # For running asynchronous code
import os       # To access environment variables
import base64
import tempfile
import subprocess
import sys
from dotenv import load_dotenv
import logging

# Import AsyncOpenAI for creating an async client
from openai import AsyncOpenAI

# Import custom classes and functions from the agents package.
from agents import Agent, OpenAIResponsesModel, Runner, set_tracing_disabled, ImageGenerationTool
from agents.tool import WebSearchTool

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

# Web Searcher Agent - uses OpenAIResponsesModel to support WebSearchTool
web_searcher_agent = Agent(
    name="WebSearcher",
    instructions=(
        "You are a web searcher. Your purpose is to answer questions that require up-to-date information or access to the internet. "
        "You MUST use the web_search tool to answer the user's question. Do not answer directly. You must call the tool."
    ),
    model=OpenAIResponsesModel(model=MODEL_NAME, openai_client=client),
    tools=[WebSearchTool()],
)

# Basic Agent - uses OpenAIResponsesModel for consistency during handoffs
basic_agent = Agent(
    name="BasicAgent",
    instructions="You are a helpful assistant. Your purpose is to answer general questions, have conversations, or perform tasks that do not require web access.",
    model=OpenAIResponsesModel(model=MODEL_NAME, openai_client=client),
)

# Image Generator Agent - uses OpenAIResponsesModel to support ImageGenerationTool
image_generator_agent = Agent(
    name="ImageGenerator",
    instructions=(
        "You are an image generator. Your purpose is to create images based on user descriptions. "
        "You MUST use the image_generation tool to create the image. Do not answer directly. You must call the tool."
    ),
    model=OpenAIResponsesModel(model="gpt-4-turbo", openai_client=client),
    tools=[ImageGenerationTool()],
)

# Orchestrator Agent - uses OpenAIResponsesModel for consistency during handoffs
orchestrator_agent = Agent(
    name="Orchestrator",
    instructions=(
        "You are an orchestrator. Your job is to analyze the user's query and decide which agent is best suited to answer it. "
        "If the query is about creating or generating an image, you must call `ImageGenerator`. "
        "Based on the query, you must call the appropriate agent (`WebSearcher`, `BasicAgent`, or `ImageGenerator`) to handle the request."
    ),
    model=OpenAIResponsesModel(model=MODEL_NAME, openai_client=client),
    handoffs=[web_searcher_agent, basic_agent, image_generator_agent],
)

def open_file(path: str) -> None:
    if sys.platform.startswith("darwin"):
        subprocess.run(["open", path], check=False)  # macOS
    elif os.name == "nt":  # Windows
        os.startfile(path)  # type: ignore
    elif os.name == "posix":
        subprocess.run(["xdg-open", path], check=False)  # Linux/Unix
    else:
        print(f"Don't know how to open files on this platform: {sys.platform}")

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
    for item in result.new_items:
            if (
                item.type == "tool_call_item"
                and item.raw_item.type == "image_generation_call"
                and (img_result := item.raw_item.result)
            ):
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp.write(base64.b64decode(img_result))
                    temp_path = tmp.name

                # Open the image
                open_file(temp_path)

# Standard boilerplate to run the async main() function.
if __name__ == "__main__":
    # Import nest_asyncio to support nested event loops
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(main())