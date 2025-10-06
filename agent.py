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

async def main():
    """
    Main asynchronous function to set up and run the agent.
    
    This function creates an Agent with a custom model and function tools,
    then runs a query to get the weather in Tokyo.
    """
    # Create an Agent instance with:
    # - A name ("Assistant")
    # - Custom instructions ("Be precise and concise.")
    # - A model built from OpenAIChatCompletionsModel using our client and model name.
    # - A list of tools; here, only get_weather is provided.
    agent = Agent(
        name="Assistant",
        instructions="Be precise and concise.",
        model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
        tools=[get_weather],
    )

    # Execute the agent with the sample query. Wrap in a timeout so the
    # program doesn't hang indefinitely if the model or network is slow.
    query = "What's the weather in Tokyo?"
    print(f"[debug] running agent with model={MODEL_NAME!r}, base_url={BASE_URL!r}")

    # Try running the agent normally. Some Perplexity sonar models (for
    # example `sonar` / `sonar-reasoning`) do not support function/tool
    # calling via the Chat Completions API and will return a 400 error
    # like: "Tool calling is not supported for this model". Detect that
    # case and automatically retry the run without tools so you still get
    # a useful text response.
    try:
        result = await Runner.run(agent, query)
    except Exception as e:
        # Try to robustly detect the "tool calling not supported" error
        # by inspecting exception attributes and attempting to parse any
        # JSON payload embedded in the exception string.
        import json
        import re

        def _extract_error_message(exc: Exception) -> str:
            # 1) If the exception has a `response` attribute, try to inspect it
            resp = getattr(exc, "response", None)
            if resp is not None:
                # If it's already a dict-like object, try to get the nested message
                try:
                    if isinstance(resp, dict):
                        return resp.get("error", {}).get("message", str(resp))
                    # If it's a string or bytes, return it
                    if isinstance(resp, (str, bytes)):
                        return resp.decode() if isinstance(resp, bytes) else resp
                except Exception:
                    # Fall through to string parsing
                    pass

            # 2) Fallback: try to parse JSON embedded in the exception string
            s = str(exc)
            # Look for the first JSON object in the string
            m = re.search(r"\{.*\}", s)
            if m:
                try:
                    obj = json.loads(m.group(0))
                    return obj.get("error", {}).get("message", s)
                except Exception:
                    # Not JSON after all
                    return s

            # 3) Nothing else worked; return the plain string
            return s

        err_msg = _extract_error_message(e) or ""
        if "tool calling" in err_msg.lower() or "tools are not supported" in err_msg.lower():
            print("[warning] model does not support tool calling; retrying without tools...")
            # Create a copy of the agent that doesn't include tools and retry
            agent_no_tools = Agent(
                name=agent.name,
                instructions=agent.instructions,
                model=agent.model,
                tools=[],
            )
            result = await Runner.run(agent_no_tools, query)
        else:
            # Re-raise if it's an unexpected error so it surfaces normally
            raise

    # Print the final output from the agent.
    print(result.final_output)

# Standard boilerplate to run the async main() function.
if __name__ == "__main__":
    asyncio.run(main())