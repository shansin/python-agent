from mcp.server.fastmcp import FastMCP
from utils import push_notification

mcp = FastMCP("push_server")

@mcp.tool()
async def push_notification_server(message: str) -> str:
    """This tool sends a push notification to the user.

    Args:
        message: the message to send
    """
    print(f"MCP Push Notification: {message}")
    push_notification(message)
    return "success"

if __name__ == "__main__":
    mcp.run(transport='stdio')