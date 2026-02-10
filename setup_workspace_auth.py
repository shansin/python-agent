import subprocess
import json
import os
import sys
import time

def main():
    workspace_dir = "/home/shant/git_linux/workspace/workspace-server"
    script_path = os.path.join(workspace_dir, "dist", "index.js")
    
    print(f"Starting MCP server at {script_path}...")
    
    # Start the Node.js MCP server
    process = subprocess.Popen(
        ["node", script_path],
        cwd=workspace_dir,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr, # Pass stderr to console so we see logs (including auth URL)
        text=True,
        bufsize=0 # Unbuffered
    )

    try:
        # 1. Initialize
        print("Sending initialize request...")
        init_req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "auth-setup", "version": "1.0"}
            }
        }
        process.stdin.write(json.dumps(init_req) + "\n")
        process.stdin.flush()
        
        # Read init response
        while True:
            line = process.stdout.readline()
            if not line: break
            try:
                msg = json.loads(line)
                if msg.get("id") == 1:
                    print("Initialized successfully.")
                    break
            except ValueError:
                continue

        # 2. Send initialized notification
        process.stdin.write(json.dumps({
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }) + "\n")
        process.stdin.flush()

        # 3. Call a tool to trigger auth
        print("\nTriggering authentication flow by calling 'calendar_list'...")
        print("Please watch the output below for an Authentication URL.\n")
        
        tool_call = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "calendar_list", # Normalized name (dots to underscores)
                "arguments": {}
            }
        }
        process.stdin.write(json.dumps(tool_call) + "\n")
        process.stdin.flush()

        # Read responses until we get the result
        while True:
            line = process.stdout.readline()
            if not line: break
            try:
                msg = json.loads(line)
                # Check for logging messages (which might contain the URL if not on stderr)
                if msg.get("method") == "notifications/message":
                    print(f"LOG: {msg['params']['data']}")
                
                if msg.get("id") == 2:
                    if "error" in msg:
                        print(f"Tool error: {msg['error']}")
                    else:
                        print("Tool execution successful! Authentication should be complete.")
                    break
            except ValueError:
                print(f"RAW: {line.strip()}")
                continue
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        process.terminate()

if __name__ == "__main__":
    main()
