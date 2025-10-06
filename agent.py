# Import necessary standard libraries
import asyncio  # For running asynchronous code
import os       # To access environment variables
from dotenv import load_dotenv

# Import AsyncOpenAI for creating an async client
from openai import AsyncOpenAI

# Import custom classes and functions from the agents package.
# These handle agent creation, model interfacing, running agents, and more.
from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool, set_tracing_disabled

# Load local .env file (if present) then retrieve configuration from environment variables
load_dotenv()

# Retrieve configuration from environment variables or use defaults
BASE_URL = os.getenv("BASE_URL") or "https://api.perplexity.ai"
API_KEY = os.getenv("PERPLEXITY_API_KEY") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME") or "sonar-pro"

# Validate that all required configuration variables are set
if not BASE_URL or not API_KEY or not MODEL_NAME:
    raise ValueError(
        "Please set PERPLEXITY_API_KEY or OPENAI_API_KEY."
    )

# Initialize the custom OpenAI async client with the specified BASE_URL and API_KEY.
client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

# Disable tracing to avoid using a platform tracing key; adjust as needed.
set_tracing_disabled(disabled=True)

# Define a function tool that the agent can call.
# The decorator registers this function as a tool in the agents framework.
@function_tool
def get_weather(city: str):
    """
    Simulate fetching weather data for a given city.
    
    Args:
        city (str): The name of the city to retrieve weather for.
        
    Returns:
        str: A message with weather information.
    """
    print(f"[debug] getting weather for {city}")
    return f"The weather in {city} is sunny."

# Import nest_asyncio to support nested event loops
import nest_asyncio

# Apply the nest_asyncio patch to enable running asyncio.run()
# even if an event loop is already running.
nest_asyncio.apply()

# Models that are known to not support tool calling
MODELS_WITHOUT_TOOL_CALLING = [
    "sonar-pro",
    "sonar-8x7b-online",
    "sonar-32k-online",
]

async def main():
    """
    Main asynchronous function to set up and run the agent.
    
    This function creates an Agent with a custom model and function tools,
    then runs a query to get the weather in Tokyo.
    """
    # Conditionally select tools based on the model's capabilities.
    # This is a more robust method than reactive error handling.
    tools = []
    if MODEL_NAME not in MODELS_WITHOUT_TOOL_CALLING:
        tools.append(get_weather)
    else:
        print(f"[info] model {MODEL_NAME!r} does not support tool calling; running without tools.")

    # Create an Agent instance with:
    # - A name ("Assistant")
    # - Custom instructions ("Be precise and concise.")
    # - A model built from OpenAIChatCompletionsModel using our client and model name.
    # - A list of tools (which may be empty).
    agent = Agent(
        name="Assistant",
        instructions="Be precise and concise.",
        model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
        tools=tools,
    )

    # Execute the agent with the sample query.
    query = "What's the weather in Tokyo?"
    print(f"[debug] running agent with model={MODEL_NAME!r}, base_url={BASE_URL!r}")
    result = await Runner.run(agent, query)

    # Print the final output from the agent.
    print(result.final_output)

# Standard boilerplate to run the async main() function.
if __name__ == "__main__":
    asyncio.run(main())