from dotenv import load_dotenv
import os
import asyncio
import json
from typing import Dict
from pydantic import BaseModel, Field
import anthropic
from claude_agent_sdk import (
    query, ClaudeSDKClient, ClaudeAgentOptions, AgentDefinition,
    ResultMessage, AssistantMessage, TextBlock, SystemMessage,
    CLINotFoundError, CLIConnectionError,
)
from utils import push_notification, send_email_sendgrid, send_email_resend, send_html_email, searxng_search, tavily_search, google_sheets_get_col

load_dotenv(override=True)

MODEL = "claude-opus-4-6"


# ── 1. Simple chat completion ─────────────────────────────────────────────────
def chat_completion(question: str):
    """Equivalent of main.py::chat_completion — direct anthropic SDK call."""
    client = anthropic.Anthropic()
    print(f"Chat completion example, question: {question}")
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": question}],
    )
    print(response.content[0].text)


# ── 2. Simple agent ───────────────────────────────────────────────────────────
async def simple_agent():
    """Equivalent of main.py::simple_agent — query() with system_prompt."""
    print("simple_agent")
    async for message in query(
        prompt="Tell me a joke about Autonomous AI agents",
        options=ClaudeAgentOptions(
            system_prompt="You are a joke teller.",
            allowed_tools=[],
            model=MODEL,
        ),
    ):
        if isinstance(message, ResultMessage):
            print(message.result)


# ── 3. Simple agent streaming ─────────────────────────────────────────────────
async def simple_agent_streaming():
    """Equivalent of main.py::simple_agent_streaming — ClaudeSDKClient streaming."""
    print("simple_agent_streaming")
    options = ClaudeAgentOptions(
        system_prompt="You are a joke teller.",
        allowed_tools=[],
        model=MODEL,
    )
    async with ClaudeSDKClient(options=options) as client:
        await client.query("Tell me a joke about Autonomous AI agents")
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(block.text, end="", flush=True)
    print()


# ── 4. Multiple agents (parallel) ─────────────────────────────────────────────
async def multiple_agents():
    """Equivalent of main.py::multiple_agents — parallel queries + picker agent."""
    print("multiple_agents")

    instructions = [
        "You are a sales agent working for ShanUP 3D, a company that provides 3D printing solutions powered by AI. You write professional, serious cold emails.",
        "You are a humorous, engaging sales agent working for ShanUP 3D, a company that provides 3D printing solutions powered by AI. You write witty, engaging cold emails that are likely to get a response.",
        "You are a busy sales agent working for ShanUP 3D, a company that provides 3D printing solutions powered by AI. You write concise, to the point cold emails.",
    ]

    prompt = "Write a cold email to a CTO of a mid-sized tech company introducing ShanUP (a 3D printing solutions company) and its benefits."

    async def run_agent(system_prompt: str) -> str:
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                system_prompt=system_prompt,
                allowed_tools=[],
                model=MODEL,
            ),
        ):
            if isinstance(message, ResultMessage):
                return message.result
        return ""

    outputs = await asyncio.gather(*[run_agent(inst) for inst in instructions])

    for output in outputs:
        print(output + "\n\n")

    combined_prompt = "Here are some cold sales email options:\n\n"
    for i, output in enumerate(outputs):
        combined_prompt += f"Option {i + 1}:\n{output}\n\n"
    combined_prompt += "Which option would you respond to? Reply with the selected email only."

    picker_instructions = (
        "You pick the best cold sales email from the given options. "
        "Imagine you are a customer and pick the one you are most likely to respond to. "
        "Do not give an explanation; reply with the selected email only."
    )

    async for message in query(
        prompt=combined_prompt,
        options=ClaudeAgentOptions(
            system_prompt=picker_instructions,
            allowed_tools=[],
            model=MODEL,
        ),
    ):
        if isinstance(message, ResultMessage):
            print("Best cold email:\n")
            print(message.result)


# ── 5. Multiple agents as tool (subagents) ────────────────────────────────────
async def multiple_agents_as_tool():
    """Equivalent of main.py::multiple_agents_as_tool — AgentDefinition subagents."""
    print("multiple_agents_as_tool")

    sales_agents = {
        "professional_sales_agent": AgentDefinition(
            description="Write professional, serious cold sales emails for ShanUP 3D.",
            prompt="You are a sales agent working for ShanUP 3D. You write professional, serious cold emails.",
            tools=[],
        ),
        "humorous_sales_agent": AgentDefinition(
            description="Write witty, engaging cold sales emails for ShanUP 3D.",
            prompt="You are a humorous, engaging sales agent working for ShanUP 3D. You write witty, engaging cold emails that are likely to get a response.",
            tools=[],
        ),
        "concise_sales_agent": AgentDefinition(
            description="Write concise, to-the-point cold sales emails for ShanUP 3D.",
            prompt="You are a busy sales agent working for ShanUP 3D. You write concise, to the point cold emails.",
            tools=[],
        ),
    }

    manager_instructions = """
You are a Sales Manager at ShanUP 3D Solutions. Your goal is to find the single best cold sales email using the available subagents.

Follow these steps:
1. Generate Drafts: Use all three sales agent subagents to generate three different email drafts.
2. Evaluate and Select: Review the drafts and choose the single best email.
3. Output ONLY the winning email body — nothing else.

Rules:
- You MUST use the sales agent subagents to generate the drafts — do not write them yourself.
- Return exactly ONE email.
"""

    best_email = ""
    async for message in query(
        prompt="Write a cold email to a CTO of a mid-sized tech company",
        options=ClaudeAgentOptions(
            system_prompt=manager_instructions,
            allowed_tools=["Agent"],
            agents=sales_agents,
            model=MODEL,
        ),
    ):
        if isinstance(message, ResultMessage):
            best_email = message.result
            print(message.result)

    if best_email:
        send_email_sendgrid(
            to="mailme.shantanu@gmail.com",
            sub="Cold Sales Email",
            body=best_email,
            type="text/plain",
        )


# ── Helper: input guardrail ───────────────────────────────────────────────────
def _check_for_name(message: str) -> bool:
    """Equivalent of main.py's @input_guardrail — uses anthropic SDK for structured check."""

    class NameCheckOutput(BaseModel):
        is_name_in_message: bool
        name: str

    client = anthropic.Anthropic()
    response = client.messages.parse(
        model=MODEL,
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": f"Check if the following message includes someone's personal name: {message}",
        }],
        output_format=NameCheckOutput,
    )
    return response.parsed_output.is_name_in_message


# ── 6. Multiple agents + handoff + guardrail ──────────────────────────────────
async def multiple_agents_as_tool_and_handoff_and_guardrail():
    """Equivalent of main.py::multiple_agents_as_tool_and_handoff_and_guardrail.

    Guardrail  → pre-flight name check using the anthropic SDK.
    Agents as tools → AgentDefinition subagents (sales writers + emailer).
    Handoff    → manager passes winning draft to emailer subagent.
    """
    print("multiple_agents_as_tool_and_handoff_and_guardrail")

    message = "Send out a cold sales email addressed to Dear CEO from Head of Business Development"

    # Guardrail: abort if a personal name is detected
    if _check_for_name(message):
        print("Guardrail triggered: personal name detected in message. Aborting.")
        return

    sales_agents = {
        "professional_sales_agent": AgentDefinition(
            description="Write professional, serious cold sales emails for ShanUP 3D.",
            prompt="You are a sales agent working for ShanUP 3D. You write professional, serious cold emails.",
            tools=[],
        ),
        "humorous_sales_agent": AgentDefinition(
            description="Write witty, engaging cold sales emails for ShanUP 3D.",
            prompt="You are a humorous, engaging sales agent working for ShanUP 3D. You write witty, engaging cold emails.",
            tools=[],
        ),
        "concise_sales_agent": AgentDefinition(
            description="Write concise, to-the-point cold sales emails for ShanUP 3D.",
            prompt="You are a busy sales agent working for ShanUP 3D. You write concise, to the point cold emails.",
            tools=[],
        ),
        "email_manager": AgentDefinition(
            description="Format an email body: write a subject line and convert to HTML.",
            prompt="""You are an email formatter. You receive an email body.
1. Write a compelling subject line.
2. Convert the body to clean, well-presented HTML.
Return your response as JSON only, with keys "subject" and "html_body".""",
            tools=[],
        ),
    }

    manager_instructions = """
You are a Sales Manager at ShanUP 3D solutions. Your goal is to find the single best cold sales email and get it formatted for sending.

Follow these steps:
1. Generate Drafts: Use all three sales agent subagents to generate three different email drafts.
2. Evaluate and Select: Choose the single best email.
3. Handoff: Pass ONLY the winning draft to the email_manager subagent to get a subject and HTML body.
4. Output the email_manager's JSON response verbatim.

Rules:
- Use the sales agent subagents to generate drafts — do not write them yourself.
- Hand off exactly ONE email to email_manager.
"""

    result = ""
    async for msg in query(
        prompt=message,
        options=ClaudeAgentOptions(
            system_prompt=manager_instructions,
            allowed_tools=["Agent"],
            agents=sales_agents,
            model=MODEL,
        ),
    ):
        if isinstance(msg, ResultMessage):
            result = msg.result
            print(result)

    # Parse the emailer's JSON output and send
    if result:
        try:
            email_data = json.loads(result)
            send_email_sendgrid(
                to="mailme.shantanu@gmail.com",
                sub=email_data["subject"],
                body=email_data["html_body"],
                type="text/html",
            )
        except (json.JSONDecodeError, KeyError):
            send_email_sendgrid(
                to="mailme.shantanu@gmail.com",
                sub="Cold Sales Email",
                body=result,
                type="text/plain",
            )


# ── 7. Basic research agent ───────────────────────────────────────────────────
async def basic_research_agent(research: str):
    """Equivalent of main.py::basic_research_agent — uses built-in WebSearch tool."""
    print("basic_research_agent")
    async for message in query(
        prompt=research,
        options=ClaudeAgentOptions(
            system_prompt=(
                "You are a research assistant. Given a search term, you search the web and "
                "produce a concise summary of the results. The summary must be 2-3 paragraphs "
                "and less than 300 words. Capture the main points. Write succinctly, no need "
                "for complete sentences or good grammar. Do not include any additional commentary "
                "other than the summary itself."
            ),
            allowed_tools=["WebSearch"],
            model=MODEL,
        ),
    ):
        if isinstance(message, ResultMessage):
            print(message.result)


# ── 8. Deep research agent ────────────────────────────────────────────────────
async def deep_research_agent(research: str, breadth: int):
    """Equivalent of main.py::deep_research_agent.

    Planner and Writer use the anthropic SDK (structured output + adaptive thinking).
    Searchers use the claude_agent_sdk with built-in WebSearch.
    Emailer uses the anthropic SDK.
    """
    print("deep_research_agent")

    client = anthropic.Anthropic()

    # ── Planner ──
    class WebSearchItem(BaseModel):
        reason: str = Field(description="Your reasoning for why this search is important.")
        query: str = Field(description="The search term to use.")

    class WebSearchPlan(BaseModel):
        searches: list[WebSearchItem] = Field(description="A list of web searches to perform.")

    print("Planning searches...")
    plan_response = client.messages.parse(
        model=MODEL,
        max_tokens=2048,
        system=f"You are a helpful research assistant. Given a query, come up with a set of web searches to perform to best answer the query. Output exactly {breadth} search terms.",
        messages=[{"role": "user", "content": f"Query: {research}"}],
        output_format=WebSearchPlan,
    )
    search_plan = plan_response.parsed_output
    print(f"Will perform {len(search_plan.searches)} searches")

    # ── Searchers (Agent SDK with WebSearch) ──
    search_instructions = (
        "You are a research assistant. Given a search term, you search the web and produce a "
        "concise summary of the results. The summary must be 2-3 paragraphs and less than 300 "
        "words. Capture the main points. Write succinctly. Do not include any additional commentary."
    )

    async def search(item: WebSearchItem) -> str:
        async for message in query(
            prompt=f"Search term: {item.query}\nReason for searching: {item.reason}",
            options=ClaudeAgentOptions(
                system_prompt=search_instructions,
                allowed_tools=["WebSearch"],
                model=MODEL,
            ),
        ):
            if isinstance(message, ResultMessage):
                return message.result
        return ""

    print("Searching...")
    search_results = await asyncio.gather(*[search(item) for item in search_plan.searches])
    print("Finished searching")

    # ── Writer (anthropic SDK with adaptive thinking) ──
    class ReportData(BaseModel):
        short_summary: str = Field(description="A short 2-3 sentence summary of the findings.")
        markdown_report: str = Field(description="The final report.")
        follow_up_questions: list[str] = Field(description="Suggested topics to research further.")

    print("Thinking about report...")
    writer_response = client.messages.parse(
        model=MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=(
            "You are a senior researcher tasked with writing a cohesive report for a research query. "
            "You will be provided with the original query and summarized search results. "
            "First come up with an outline, then generate the report. "
            "The final output should be in markdown format, lengthy and detailed. Aim for 5-10 pages, at least 1000 words."
        ),
        messages=[{
            "role": "user",
            "content": f"Original query: {research}\nSummarized search results: {search_results}",
        }],
        output_format=ReportData,
    )
    report = writer_response.parsed_output
    print("Finished writing report")

    # ── Emailer (anthropic SDK) ──
    print("Writing email...")
    email_response = client.messages.create(
        model=MODEL,
        max_tokens=8000,
        system=(
            "You are able to send a nicely formatted HTML email based on a detailed report. "
            "Convert the report to clean, well-presented HTML. Return ONLY the HTML — no other text."
        ),
        messages=[{"role": "user", "content": report.markdown_report}],
    )
    html_body = email_response.content[0].text
    send_email_sendgrid(
        to="mailme.shantanu@gmail.com",
        sub=f"Research Report: {research[:50]}",
        body=html_body,
        type="text/html",
    )
    print("Email sent")
    print("Hooray!")


# ── 9. MCP server (push + brave search + filesystem) ─────────────────────────
async def mcp_server(query_text: str):
    """Equivalent of main.py::mcp_server — MCP servers via mcp_servers dict."""
    print("mcp_server")

    sandbox_path = os.path.abspath(os.path.join(os.getcwd(), "sandbox"))

    async for message in query(
        prompt=query_text,
        options=ClaudeAgentOptions(
            system_prompt=(
                "You are able to search the web for information and briefly summarize the takeaways. "
                "Search the web, summarize the takeaways in a research.md file, and send a push "
                "notification to the user when research is completed."
            ),
            allowed_tools=["Write"],
            mcp_servers={
                "push": {"command": "uv", "args": ["run", "push_mcp_server.py"]},
                "files": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", sandbox_path],
                },
                "brave": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-brave-search"],
                    "env": {"BRAVE_API_KEY": os.getenv("BRAVE_API_KEY", "")},
                },
            },
            model=MODEL,
        ),
    ):
        if isinstance(message, ResultMessage):
            print(message.result)


# ── 10. WhatsApp MCP server ───────────────────────────────────────────────────
async def whatsapp_mcp_server(query_text: str):
    """Equivalent of main.py::whatsapp_mcp_server."""
    print("whatsapp_mcp_server")

    from datetime import datetime
    from zoneinfo import ZoneInfo

    now = datetime.now(ZoneInfo("America/Los_Angeles"))
    current_time = now.strftime("%I:%M %p PST, %B %d, %Y")

    async for message in query(
        prompt=query_text,
        options=ClaudeAgentOptions(
            system_prompt=(
                f"Date and time right now is {current_time}. "
                "You are able to interact with WhatsApp through the WhatsApp MCP server. "
                "Use this to fulfill user requests. Do not ask follow-up questions. Just do the task and finish."
            ),
            mcp_servers={
                "whatsapp": {
                    "command": "uv",
                    "args": [
                        "--directory",
                        "/home/shsin/git_linux/whatsapp-mcp/whatsapp-mcp-server",
                        "run",
                        "main.py",
                    ],
                }
            },
            model=MODEL,
            max_turns=30,
        ),
    ):
        if isinstance(message, ResultMessage):
            print(message.result)


# ── 11. Google Sheets MCP interactive ────────────────────────────────────────
async def google_sheets_mcp_interactive():
    """Equivalent of main.py::google_sheets_mcp_interactive."""
    print("google_sheets_mcp_interactive")

    from datetime import datetime
    from zoneinfo import ZoneInfo

    now = datetime.now(ZoneInfo("America/Los_Angeles"))
    current_time = now.strftime("%I:%M %p PST, %B %d, %Y")

    google_sheets_mcp = {
        "command": "uvx",
        "args": ["mcp-google-sheets@latest"],
        "env": {
            "GOOGLE_APPLICATION_CREDENTIALS": "./service-account.json",
            "DRIVE_FOLDER_ID": "1GMR98gr02rkum5fPWvs94AtZCcS-Lrm9",
        },
    }

    system_prompt = f"""Date and time right now is {current_time}.

You are a polite and professional restaurant manager. Your job is to take customer's order for food.

You have access to a Google Sheet with Inventory, Orders and FAQs tabs:
- Inventory: Lists food items with prices, quantity available, and description.
- Orders: Lists all orders placed by customers. Update this tab and inventory quantity when a new order is placed. Customer's phone number is in the "sender" field — don't ask for it.
- FAQs: Frequently asked questions and their answers.

Use this to help users fulfill their orders."""

    print("\nGoogle Sheets Interactive (type 'exit' to quit)\n")
    while True:
        user_query = input("Enter your query: ")
        if user_query.lower() == "exit":
            break

        async for message in query(
            prompt=user_query,
            options=ClaudeAgentOptions(
                system_prompt=system_prompt,
                mcp_servers={"google_sheets": google_sheets_mcp},
                model=MODEL,
                max_turns=20,
            ),
        ):
            if isinstance(message, ResultMessage):
                print(message.result)


# ── 12. Google Workspace interactive ─────────────────────────────────────────
async def google_interactive():
    """Equivalent of main.py::google_interactive."""
    print("google_interactive")

    from datetime import datetime
    from zoneinfo import ZoneInfo

    now = datetime.now(ZoneInfo("America/Los_Angeles"))
    current_time = now.strftime("%I:%M %p PST, %B %d, %Y")

    workspace_server_path = "/home/shsin/git_linux/workspace/workspace-server/dist/index.js"

    system_prompt = f"""Date and time right now is {current_time}.

You are a powerful Google Workspace assistant with access to the user's Google Account.

You can help with:
- Google Docs: Create, read, find, update, and move documents.
- Google Drive: Find/create folders, search for files, download files.
- Google Calendar: List calendars, view/create/update/delete events, find free time.
- Google Sheets: Read content, get ranges, find spreadsheets.
- Google Slides: Read text, find presentations.
- Gmail: Search threads, draft/send emails, manage labels.
- Google Chat: List spaces, send messages, send DMs.

Behavioral rules:
1. Always respect the user's timezone ({now.tzinfo}).
2. PREVIEW write operations before executing. Ask for confirmation on destructive actions.
3. Use search tools to find IDs — do not hallucinate them.
4. Be concise, helpful, and professional."""

    workspace_mcp = {
        "command": "node",
        "args": [workspace_server_path, "--use-dot-names"],
        "env": {
            "GEMINI_CLI_WORKSPACE_FORCE_FILE_STORAGE": "true",
            "PATH": os.environ["PATH"],
        },
    }

    print("\nGoogle Workspace Interactive Assistant (type 'exit' to quit)\n")
    while True:
        user_query = input("Enter your query: ")
        if user_query.lower() == "exit":
            break

        async for message in query(
            prompt=user_query,
            options=ClaudeAgentOptions(
                system_prompt=system_prompt,
                mcp_servers={"workspace": workspace_mcp},
                model=MODEL,
                max_turns=30,
            ),
        ):
            if isinstance(message, ResultMessage):
                print(f"\n{message.result}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    push_notification("Starting Claude agent examples")
    send_email_sendgrid(
        to="mailme.shantanu@gmail.com",
        sub="Starting Claude agent examples",
        body="The Claude agent examples script has started running.",
        type="text/plain",
    )

    chat_completion("Write a story about a cart")
    await simple_agent()
    await simple_agent_streaming()
    await multiple_agents()
    await multiple_agents_as_tool()
    await multiple_agents_as_tool_and_handoff_and_guardrail()
    await basic_research_agent("Top Agentic AI frameworks to look forward to in 2026")
    await deep_research_agent("Top Agentic AI frameworks to look forward to in 2026", 3)
    await mcp_server("Top Agentic AI frameworks to look forward to in 2026")
    await whatsapp_mcp_server("get all messages sent in last 24 hours")
    await google_sheets_get_col(
        "https://docs.google.com/spreadsheets/d/17xyB3frdJsuJLTpBxYYXT1Mtr_edLOhOAL3dJHzJrTo/edit?gid=0#gid=0",
        "A",
        "FAQ",
    )
    # await google_sheets_mcp_interactive()
    # await google_interactive()


if __name__ == "__main__":
    asyncio.run(main())
