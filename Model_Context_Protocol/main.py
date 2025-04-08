import os
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Test")

@mcp.tool()
def get_os() -> str:
    """
    Get the current operating system.
    """
    return os.uname().sysname

if __name__ == "__main__":
    mcp.run(transport="stdio")