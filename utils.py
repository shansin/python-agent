import os
import requests
import json
from typing import Dict, List
from agents import function_tool
import sendgrid
from sendgrid.helpers.mail import Mail, Email, To, Content
from tavily import TavilyClient

def push_notification(message):
    import requests
    #pushover setup
    pushover_user = os.getenv("PUSHOVER_USER")
    pushover_token = os.getenv("PUSHOVER_TOKEN")
    pushover_url = "https://api.pushover.net/1/messages.json"
    print(f"Push notificaiton to phone: {message}")
    payload = {"user": pushover_user, "token": pushover_token, "message": message}
    requests.post(pushover_url, data=payload)

def send_email_sendgrid(to: str, sub: str, body: str, type: str) -> dict:
    print(f"Sending email using SendGrid to {to}, subject: {sub}")
    sg = sendgrid.SendGridAPIClient(api_key=os.environ.get('SENDGRID_API_KEY'))

    from_email = Email("mailme.shantanu@gmail.com")
    to_email = To(to)
    content = Content(type, body)
    mail = Mail(from_email, to_email, sub, content).get()
    sg.client.mail.send.post(request_body=mail)
    return {"status": "success"}

def send_email_resend(to: List[str], sub: str, from_name: str, from_email: str, body: str) -> dict:
    print(f"Sending email using Resend to {to}, subject: {sub}")
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

@function_tool
def send_html_email(subject: str, html_body: str) -> Dict[str, str]:
    """ Send out an email with the given subject and HTML body to all sales prospects """
    send_email_sendgrid(to="mailme.shantanu@gmail.com", sub="Cold Sales Email", body=html_body, type="text/html")
    return {"status": "success"}

def searxng_search(search: str, page_no: int):
    # import requests # already imported at top
    # import json # already imported at top
    # from pprint import pprint
    print(f"Searching using SearxNG for {search} on page {page_no}")
    endpoint = f"{os.getenv('SEARXNG_API_URL')}/search"

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
    #print(json.dumps(results, indent=4))
    return results

def tavily_search(search: str, max_results: int):
    print(f"Searching using Tavily for {search} with max results {max_results}")
    # from tavily import TavilyClient # already imported at top
    # import json

    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    search_results = tavily_client.search(
        query=search, 
        max_results=max_results,
        time_range="week",
        include_raw_content=True,
        #include_domains=["techcrunch.com"],
        topic="news")
        
    #print(json.dumps(search_results, indent=4))
    return search_results
