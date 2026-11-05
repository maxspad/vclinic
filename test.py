"""Smoke test for the OpenAI connection.

Run `python test.py` after filling in .env to confirm the API key, base URL, and
model are all working before starting the Streamlit app.
"""

from openai import AsyncOpenAI
from agents import set_default_openai_api, set_default_openai_client, Agent, Runner, set_tracing_disabled
from dotenv import load_dotenv
import os
import asyncio

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL')
OPENAI_ORG_ID = os.getenv('OPENAI_ORG_ID')
OPENAI_PROJECT = os.getenv('OPENAI_PROJECT')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4.1')
OPENAI_API_MODE = os.getenv('OPENAI_API_MODE', 'chat_completions')

if not OPENAI_API_KEY:
    raise SystemExit('OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.')

# Create the OpenAI client
openai_client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    organization=OPENAI_ORG_ID,
    project=OPENAI_PROJECT,
)

set_default_openai_client(openai_client, use_for_tracing=False)
set_default_openai_api(OPENAI_API_MODE)
set_tracing_disabled(disabled=True)


async def main():
    print(f'base_url: {openai_client.base_url}')
    print(f'model:    {OPENAI_MODEL}')
    print(f'api mode: {OPENAI_API_MODE}')

    agent = Agent(
        name="Assistant",
        instructions="You only respond in haikus.",
        model=OPENAI_MODEL,
    )

    result = await Runner.run(agent, "Tell me about recursion in programming.")
    print(result.final_output)
    # Function calls itself,
    # Looping in smaller pieces,
    # Endless by design.


if __name__ == "__main__":
    asyncio.run(main())
