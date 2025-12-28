from email.mime import message
from dotenv import load_dotenv
from openai import OpenAI
import os
from agents import Agent, Runner, trace, OpenAIChatCompletionsModel, input_guardrail, GuardrailFunctionOutput
from pydantic import BaseModel
from openai import AsyncOpenAI
from agents import function_tool
from typing import Dict
import asyncio

load_dotenv(override=True)

ollama_base_url = os.getenv("OLLAMA_BASE_URL")

#push notification function
def push_notification(message):
    import requests
    #pushover setup
    pushover_user = os.getenv("PUSHOVER_USER")
    pushover_token = os.getenv("PUSHOVER_TOKEN")
    pushover_url = "https://api.pushover.net/1/messages.json"
    print(f"Push notificaiton to phone: {message}")
    payload = {"user": pushover_user, "token": pushover_token, "message": message}
    requests.post(pushover_url, data=payload)

#simple chat completion example
def chat_completion_example():
     openai = OpenAI(base_url=ollama_base_url, api_key="ollama")
     question = "Write a story about a lonely computer in the style of Dr. Seuss."
     
     response = openai.chat.completions.create(
         model="gpt-oss:20b",
         messages=[{"role": "user", "content": question}],
         temperature=0.7,)

     print(response.choices[0].message.content)

#agents example (simple)
async def simple_agent_example():
    client = AsyncOpenAI(base_url= ollama_base_url, api_key="ollama")
    llama_model = OpenAIChatCompletionsModel(model="llama3.1:8b", openai_client=client)
    agent = Agent(name="JokePusher", instructions="You are a joke teller.", model=llama_model)

    with trace("Telling a joke"):
        result = await Runner.run(agent, "Tell me a joke about Autonomous AI agents")
        print(result.final_output)

def send_email(to: str, sub: str, body: str, type: str) -> dict:
    import sendgrid
    from sendgrid.helpers.mail import Mail, Email, To, Content

    sg = sendgrid.SendGridAPIClient(api_key=os.environ.get('SENDGRID_API_KEY'))

    from_email = Email("mailme.shantanu@gmail.com")
    to_email = To(to)
    content = Content(type, body)
    mail = Mail(from_email, to_email, sub, content).get()
    sg.client.mail.send.post(request_body=mail)
    return {"status": "success"}

def send_email_resend(to: [str], sub: str, from_name: str, from_email: str, body: str) -> dict:
    import requests
    # from_email has to be configured on dns, xxx@shanup.com is enabled
    headers = {
        "Authorization": f"Bearer {os.environ.get('RESEND_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "from": f"{from_name} <{from_email}>",
        "to": to,
        "subject": sub,
        "html": f"<p>{body}</p>"  # Body wrapped in <p> tags for HTML format
    }
    
    # Send email using Resend API
    response = requests.post("https://api.resend.com/emails", json=payload, headers=headers)
    
    # Check if the request was successful
    if response.ok:
        return {"status": "success"}
    else:
        return {"status": "failure", "message": response.text}

#agents example (simple streaming)
#traces at: https://platform.openai.com/logs?api=traces 
async def simple_agent_streaming_example():
    from openai.types.responses import ResponseTextDeltaEvent

    client = AsyncOpenAI(base_url=ollama_base_url, api_key="ollama")
    gpt_model = OpenAIChatCompletionsModel(model="gpt-oss:20b", openai_client=client)
    agent = Agent(name="JokePusher", instructions="You are a joke teller.", model=gpt_model)

    with trace("Telling a joke (streamed)"):
        result = Runner.run_streamed(agent, input="Tell me a joke about Autonomous AI agents")

        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta, end="", flush=True)

#agents example (multiple agents)
#traces at: https://platform.openai.com/logs?api=traces 
async def multiple_agents_example():
    client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    gpt_model = OpenAIChatCompletionsModel(model="gpt-oss:20b", openai_client=client)
    llama_model = OpenAIChatCompletionsModel(model="llama3.1:8b", openai_client=client)
    deepseek_model = OpenAIChatCompletionsModel(model="deepseek-r1:8b", openai_client=client)

    instructions1 = "You are a sales agent working for ShanUP 3D, \
    a company that provides a 3D printing solutions powered by AI. \
    You write professional, serious cold emails."

    instructions2 = "You are a humorous, engaging sales agent working for ShanUP 3D, \
    a company that provides a 3D printing solutions powered by AI. \
    You write witty, engaging cold emails that are likely to get a response."

    instructions3 = "You are a busy sales agent working for ShanUP 3D, \
    a company that provides a 3D printing solutions powered by AI. \
    You write concise, to the point cold emails."

    sales_agent1 = Agent(name="ProfessionalAgent", instructions=instructions1, model=gpt_model)
    sales_agent2 = Agent(name="HumorousAgent", instructions=instructions2, model=deepseek_model)
    sales_agent3 = Agent(name="ConciseAgent", instructions=instructions3, model=llama_model)
    
    prompt = "Write a cold email to a CTO of a mid-sized tech company introducing ShanUP (a 3D printing solutions company) and its benefits."

    with trace("Parallel cold emails and evaluation"):
        results = await asyncio.gather(
            Runner.run(sales_agent1, prompt),
            Runner.run(sales_agent2, prompt),
            Runner.run(sales_agent3, prompt))

        outputs = [result.final_output for result in results]

        for output in outputs:
            print(output + "\n\n")
    
        instructions4 = "You pick the best cold sales email from the given options. \
        Imagine you are a customer and pick the one you are most likely to respond to. \
        Do not give an explanation; reply with the selected email only."
        
        sales_picker_agent = Agent(
        name="sales_picker",
        instructions=instructions4,
        model=gpt_model)

        combined_prompt = "Here are some cold sales email options:\n\n"
        for i, output in enumerate(outputs):
            combined_prompt += f"Option {i+1}:\n{output}\n\n"
        combined_prompt += "Which option would you respond to? Reply with the selected email only."

        final_result = await Runner.run(sales_picker_agent, combined_prompt)
        print("Best cold email:\n")
        print(final_result.final_output)

#agents example (multiple agents as tool)
#traces at: https://platform.openai.com/logs?api=traces 
async def multiple_agents_as_tool_example():
    client = AsyncOpenAI(base_url=ollama_base_url, api_key="ollama")
    gpt_model = OpenAIChatCompletionsModel(model="gpt-oss:20b", openai_client=client)
    llama_model = OpenAIChatCompletionsModel(model="llama3.1:8b", openai_client=client)
    deepseek_model = OpenAIChatCompletionsModel(model="deepseek-r1:8b", openai_client=client)

    instructions1 = "You are a sales agent working for ShanUP 3D, \
    a company that provides a 3D printing solutions powered by AI. \
    You write professional, serious cold emails."

    instructions2 = "You are a humorous, engaging sales agent working for ShanUP 3D, \
    a company that provides a 3D printing solutions powered by AI. \
    You write witty, engaging cold emails that are likely to get a response."

    instructions3 = "You are a busy sales agent working for ShanUP 3D, \
    a company that provides a 3D printing solutions powered by AI. \
    You write concise, to the point cold emails."

    sales_agent1 = Agent(name="ProfessionalAgent", instructions=instructions1, model=llama_model)
    sales_agent2 = Agent(name="HumorousAgent", instructions=instructions2, model=llama_model)
    sales_agent3 = Agent(name="ConciseAgent", instructions=instructions3, model=llama_model)

    description = "Write cold sales emails"
    tool1 = sales_agent1.as_tool(tool_name="sales_agent1", tool_description=description)
    tool2 = sales_agent2.as_tool(tool_name="sales_agent2", tool_description=description)
    tool3 = sales_agent3.as_tool(tool_name="sales_agent3", tool_description=description)

    tools = [tool1, tool2, tool3, send_cold_sales_email]

    instructions = """
        You are a Sales Manager at ShanUP 3D Solutions. Your goal is to find the single best cold sales email using the sales_agent tools.

        Follow these steps carefully:
        1. Generate Drafts: Use all three sales_agent tools to generate three different email drafts. Do not proceed until all three drafts are ready.

        2. Evaluate and Select: Review the drafts and choose the single best email using your judgment of which one is most effective.

        3. Use the send_email tool to send the best email (and only the best email) to the user.

        Crucial Rules:
        - You must use the sales agent tools to generate the drafts — do not write them yourself.
        - You must send ONE email using the send_emailsend_cold_sales_email tool — never more than one.
        """

    sales_manager_agent = Agent(name="SalesManager", instructions=instructions, model=llama_model, tools=tools)
    prompt = "Write a cold email to a CTO of a mid-sized tech company"

    with trace("Sales manager using sales agents as tools"):
        results = await Runner.run(sales_manager_agent, prompt)
        print(results.final_output)

@function_tool
def send_cold_sales_email(body: str):
    """Sends a cold sales email to a predefined email address."""
    #above line acts as description for the tool to llm
    send_email(to="mailme.shantanu@gmail.com", sub="Cold Sales Email", body=body, type="text/plain")

async def multiple_agents_as_tool_and_handoff_and_guardrail_example():
    client = AsyncOpenAI(base_url=ollama_base_url, api_key="ollama")
    gpt_model = OpenAIChatCompletionsModel(model="gpt-oss:20b", openai_client=client)
    llama_model = OpenAIChatCompletionsModel(model="llama3.1:8b", openai_client=client)
    deepseek_model = OpenAIChatCompletionsModel(model="deepseek-r1:8b", openai_client=client)

    instructions1 = "You are a sales agent working for ShanUP 3D, \
    a company that provides a 3D printing solutions powered by AI. \
    You write professional, serious cold emails."

    instructions2 = "You are a humorous, engaging sales agent working for ShanUP 3D, \
    a company that provides a 3D printing solutions powered by AI. \
    You write witty, engaging cold emails that are likely to get a response."

    instructions3 = "You are a busy sales agent working for ShanUP 3D, \
    a company that provides a 3D printing solutions powered by AI. \
    You write concise, to the point cold emails."

    sales_agent1 = Agent(name="ProfessionalAgent", instructions=instructions1, model=gpt_model)
    sales_agent2 = Agent(name="HumorousAgent", instructions=instructions2, model=gpt_model)
    sales_agent3 = Agent(name="ConciseAgent", instructions=instructions3, model=gpt_model)

    description = "Write cold sales emails"
    agent_tool1 = sales_agent1.as_tool(tool_name="sales_agent1", tool_description=description)
    agent_tool2 = sales_agent2.as_tool(tool_name="sales_agent2", tool_description=description)
    agent_tool3 = sales_agent3.as_tool(tool_name="sales_agent3", tool_description=description)

    class NameCheckOutput(BaseModel):
        is_name_in_message: bool
        name: str

    guardrail_agent = Agent( 
        name="Name check",
        instructions="Check if the user is including someone's personal name in what they want you to do.",
        output_type=NameCheckOutput,
        model=gpt_model
    )

    @input_guardrail
    async def gaurdrail_against_name(ctx, agent, message):
        result = await Runner.run(guardrail_agent, message, context=ctx.context)
        is_name_in_message = result.final_output.is_name_in_message
        return GuardrailFunctionOutput(output_info={"found_name": result.final_output},tripwire_triggered= is_name_in_message)

    subject_instructions = "You can write a subject for a cold sales email. \
    You are given a message and you need to write a subject for an email that is likely to get a response."
    
    subject_writer = Agent(name="Email subject writer", instructions=subject_instructions, model=gpt_model)
    subject_tool = subject_writer.as_tool(tool_name="subject_writer", tool_description="Write a subject for a cold sales email")

    html_instructions = "You can convert a text email body to an HTML email body. \
    You are given a text email body which might have some markdown \
    and you need to convert it to an HTML email body with simple, clear, compelling layout and design."

    html_converter = Agent(name="Html email body converter", instructions=html_instructions, model=gpt_model)
    html_converter_tool = html_converter.as_tool(tool_name="html_converter",tool_description="Convert a text email body to an HTML email body")

    instructions ="You are an email formatter and sender. You receive the body of an email to be sent. \
    You first use the subject_writer tool to write a subject for the email, then use the html_converter tool to convert the body to HTML. \
    Finally, you use the send_html_email tool to send the email with the subject and HTML body."
    
    emailer_agent = Agent(
        name="Email Manager",
        instructions=instructions,
        tools=[subject_tool, html_converter_tool, send_html_email],
        model=gpt_model,
        handoff_description="Convert an email to HTML and send it")

    sales_manager_instructions = """
        You are a Sales Manager at ShanUP 3D solutions. Your goal is to find the single best cold sales email using the sales_agent tools.
 
        Follow these steps carefully:
        1. Generate Drafts: Use all three sales_agent tools to generate three different email drafts. Do not proceed until all three drafts are ready.
        
        2. Evaluate and Select: Review the drafts and choose the single best email using your judgment of which one is most effective.
        You can use the tools multiple times if you're not satisfied with the results from the first try.
        
        3. Handoff for Sending: Pass ONLY the winning email draft to the 'Email Manager' agent. The Email Manager will take care of formatting and sending.
        
        Crucial Rules:
        - You must use the sales agent tools to generate the drafts — do not write them yourself.
        - You must hand off exactly ONE email to the Email Manager — never more than one.
        """
    
    sales_manager = Agent(
        name="Sales Manager",
        instructions=sales_manager_instructions,
        tools=[agent_tool1, agent_tool2, agent_tool3],
        handoffs=[emailer_agent],
        input_guardrails=[gaurdrail_against_name],
        model=gpt_model)
    
    message = "Send out a cold sales email addressed to Dear CEO from Head of Business Development"

    with trace("Automated Sales Manager"):
        result = await Runner.run(sales_manager, message)
        print(result.final_output)

@function_tool
def send_html_email(subject: str, html_body: str) -> Dict[str, str]:
    """ Send out an email with the given subject and HTML body to all sales prospects """
    send_email(to="mailme.shantanu@gmail.com", sub="Cold Sales Email", body=html_body, type="text/html")
    return {"status": "success"}


def searxng_search(search: str, page_no: int):
    import requests
    import json
    from pprint import pprint

    endpoint = f"{os.getenv("SEARXNG_API_URL")}/search"

    params = {
        "q": search,
        "format": "json",
        "categories": "general",
        "language": "en",
        "safesearch": 0,
        "pageno": page_no
    }

    response = requests.get(endpoint, params=params, timeout=10)
    response.raise_for_status()
    results = response.json().get("results", [])
    print(json.dumps(results, indent=4))
    return results

#tutorial https://github.com/NirDiamant/agents-towards-production/blob/main/tutorials/agent-with-tavily-web-access/search-extract-crawl.ipynb
def tavily_search(search: str, max_results: int):
    from tavily import TavilyClient
    import json
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    search_results = tavily_client.search(
        query=search, 
        max_results=max_results,
        time_range="week",
        include_raw_content=True,
        #include_domains=["techcrunch.com"],
        topic="news")
        
    print(json.dumps(search_results, indent=4))

    return search_results

async def main():
    #push_notification("Starting OpenAI agent examples")
    #send_email(to="mailme.shantanu@gmail.com", sub="Starting OpenAI agent examples", body="The OpenAI agent examples script has started running.")
    #send_email_resend(to=["mailme.shantanu@gmail.com","dixit.upasanaitbhu@gmail.com"], sub="Welcome to ShanUP.com", body="Mark this date as when it started. \n<a href=\"https://www.shanup.com/\">Visit ShanUP.com!</a>", from_name="Shantanu", from_email="Shantanu@shanup.com")
    searxng_search("News in Bothell Washington", 1)
    #tavily_search("News in Bothell Washington", 20)
    #chat_completion_example()
    #await simple_agent_example()
    #await simple_agent_streaming_example()
    #await multiple_agents_example()
    #await multiple_agents_as_tool_example()
    #await multiple_agents_as_tool_and_handoff_and_guardrail_example()
    

if __name__ == "__main__":
    asyncio.run(main())