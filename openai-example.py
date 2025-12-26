from dotenv import load_dotenv
from openai import OpenAI
import requests
import os
from agents import Agent, Runner, trace
from agents import OpenAIChatCompletionsModel
from openai import AsyncOpenAI
import asyncio

load_dotenv(override=True)


#pushover example
pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "https://api.pushover.net/1/messages.json"

def push(message):
    print(f"Push: {message}")
    payload = {"user": pushover_user, "token": pushover_token, "message": message}
    requests.post(pushover_url, data=payload)

def make_message(role: str, content: str) -> list[dict[str, str]]:
    return [{"role": role, "content": content}]

openai = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

#simple chat completion example
def chat_completion_example():
     question = "Write a story about a lonely computer in the style of Dr. Seuss."
     model = "llama3.1:8b"
     print(f"--- Model: {model} ---")
     push(f"Starting request for model: {model}")

     response = openai.chat.completions.create(
         model=model,
         messages=make_message("user", question),
         temperature=0.7,)

     print(response.choices[0].message.content)

     push(f"Finished request for model: {model}")

#agents example (simple)
async def agents_simple_example():
    client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    gpt_model = OpenAIChatCompletionsModel(model="gpt-oss:20b", openai_client=client)
    agent = Agent(name="JokePusher", instructions="You are a joke teller.", model=gpt_model)

    with trace("Telling a joke"):
        result = await Runner.run(agent, "Tell me a joke about Autonomous AI agents")
        print(result.final_output)


#agents example (multiple agents)
async def agents_multiple_example():
    client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    instructions1 = "You are a sales agent working for ComplAI, \
    a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI. \
    You write professional, serious cold emails."

    instructions2 = "You are a humorous, engaging sales agent working for ComplAI, \
    a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI. \
    You write witty, engaging cold emails that are likely to get a response."

    instructions3 = "You are a busy sales agent working for ComplAI, \
    a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI. \
    You write concise, to the point cold emails."
    
    gpt_model = OpenAIChatCompletionsModel(model="gpt-oss:20b", openai_client=client)
    llama_model = OpenAIChatCompletionsModel(model="llama3.1:8b", openai_client=client)
    deepseek_model = OpenAIChatCompletionsModel(model="deepseek-r1:8b", openai_client=client)

    sales_agent1 = Agent(name="ProfessionalAgent", instructions=instructions1, model=gpt_model)
    sales_agent2 = Agent(name="HumorousAgent", instructions=instructions2, model=gpt_model)
    sales_agent3 = Agent(name="ConciseAgent", instructions=instructions3, model=gpt_model)

    prompt = "Write a cold email to a CTO of a mid-sized tech company introducing ShanUP (a 3D printing solutions company) and its benefits."

    #result = await Runner.run(sales_agent1, prompt)
    with trace("Parallel cold emails"):
        results = await asyncio.gather(
            Runner.run(sales_agent1, prompt),
            Runner.run(sales_agent2, prompt),
            Runner.run(sales_agent3, prompt))

        outputs = [result.final_output for result in results]

        for output in outputs:
            print(output + "\n\n")

async def main():
    #chat_completion_example()
    #await agents_simple_example()
    await agents_multiple_example()


if __name__ == "__main__":
    asyncio.run(main())