import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

OLLAMA_BASE_URL = "http://localhost:11434/v1"

ollama_provider = OllamaProvider(base_url=OLLAMA_BASE_URL)

model = OpenAIChatModel(
    model_name="gpt-oss:20b", 
    provider=ollama_provider
)

agent = Agent(model=model, instructions="You are a helpful assistant.")

async def main():
    result = await agent.run('What is the capital of France?')
    print(result.output)
    #> The capital of France is Paris.

    async with agent.run_stream('Write a story about a shopping cart') as response:
        async for text in response.stream_text():
            print(text)
            #> The capital of
            #> The capital of the UK is
            #> The capital of the UK is London.

if __name__ == "__main__":
    asyncio.run(main())