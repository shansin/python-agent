from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

# Set the base URL with the required /v1 API suffix
OLLAMA_BASE_URL = "http://localhost:11434/v1"

# 1. Instantiate the OllamaProvider with the custom URL
ollama_provider = OllamaProvider(base_url=OLLAMA_BASE_URL)

# 2. Instantiate the OpenAIChatModel, specifying the model name you pulled (e.g., "llama3")
# and passing the configured provider.
model = OpenAIChatModel(
    model_name="gpt-oss:20b", 
    provider=ollama_provider
)

# 3. Create your Agent with the configured model
agent = Agent(model=model, instructions="You are a helpful assistant.")

# Example usage (assuming 'run_sync' is the correct method in your version)
response = agent.run_sync("Tell me a fun fact.")
print(response.output)