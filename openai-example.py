from dotenv import load_dotenv
from openai import OpenAI
import requests
import os
from agents import Agent, Runner, trace
from agents import OpenAIChatCompletionsModel
from openai import AsyncOpenAI
import asyncio

load_dotenv(override=True)

#pushover setup
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
async def simple_agent_example():
    client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    gpt_model = OpenAIChatCompletionsModel(model="gpt-oss:20b", openai_client=client)
    agent = Agent(name="JokePusher", instructions="You are a joke teller.", model=gpt_model)

    with trace("Telling a joke"):
        result = await Runner.run(agent, "Tell me a joke about Autonomous AI agents")
        print(result.final_output)


#agents example (multiple agents)
#traces at: https://platform.openai.com/logs?api=traces 
async def multiple_agents_example():
    client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    instructions1 = "You are a sales agent working for ShanUP 3D, \
    a company that provides a 3D printing solutions powered by AI. \
    You write professional, serious cold emails."

    instructions2 = "You are a humorous, engaging sales agent working for ShanUP 3D, \
    a company that provides a 3D printing solutions powered by AI. \
    You write witty, engaging cold emails that are likely to get a response."

    instructions3 = "You are a busy sales agent working for ShanUP 3D, \
    a company that provides a 3D printing solutions powered by AI. \
    You write concise, to the point cold emails."

    instructions4 = "You pick the best cold sales email from the given options. \
    Imagine you are a customer and pick the one you are most likely to respond to. \
    Do not give an explanation; reply with the selected email only."
    
    gpt_model = OpenAIChatCompletionsModel(model="gpt-oss:20b", openai_client=client)
    llama_model = OpenAIChatCompletionsModel(model="llama3.1:8b", openai_client=client)
    deepseek_model = OpenAIChatCompletionsModel(model="deepseek-r1:8b", openai_client=client)

    sales_agent1 = Agent(name="ProfessionalAgent", instructions=instructions1, model=gpt_model)
    sales_agent2 = Agent(name="HumorousAgent", instructions=instructions2, model=gpt_model)
    sales_agent3 = Agent(name="ConciseAgent", instructions=instructions3, model=gpt_model)

    prompt = "Write a cold email to a CTO of a mid-sized tech company introducing ShanUP (a 3D printing solutions company) and its benefits."

    with trace("Parallel cold emails and evaluation"):
        results = await asyncio.gather(
            Runner.run(sales_agent1, prompt),
            Runner.run(sales_agent2, prompt),
            Runner.run(sales_agent3, prompt))

        outputs = [result.final_output for result in results]

        for output in outputs:
            print(output + "\n\n")
    
        sales_picker_agent = Agent(
        name="sales_picker",
        instructions=instructions4,
        model=llama_model)

        combined_prompt = "Here are some cold sales email options:\n\n"
        for i, output in enumerate(outputs):
            combined_prompt += f"Option {i+1}:\n{output}\n\n"
        combined_prompt += "Which option would you respond to? Reply with the selected email only."

        final_result = await Runner.run(sales_picker_agent, combined_prompt)
        print("Best cold email:\n")
        print(final_result.final_output)

async def main():
    #chat_completion_example()
    #await simple_agent_example()
    await multiple_agents_example()


if __name__ == "__main__":
    asyncio.run(main())