"""
FastMCP Tools Integration Module
Simple, general-purpose MCP client using STREAMABLE HTTP transport
Works with ANY MCP server, not just healthcare
"""
import logging
import os
from typing import Dict, List, Any
from fastmcp import Client

logger = logging.getLogger(__name__)

# MCP Server Configuration - Streamable HTTP endpoint
MCP_SERVER_URL = os.environ.get('MCP_SERVER_URL', 'https://epicmcp.cdwaws.com/mcp/')

async def get_available_tools(server_url: str = None) -> List[Dict[str, Any]]:
    """
    Get all available tools from the MCP server using Streamable HTTP transport
    Returns list of tool objects with name, description, etc.
    """
    url = server_url or MCP_SERVER_URL
    
    try:
        # FastMCP auto-detects Streamable HTTP transport from http/https URLs
        logger.info(f"🌐 Connecting to MCP server via Streamable HTTP: {url}")
        async with Client(url) as client:
            tools = await client.list_tools()
            logger.info(f"✅ Retrieved {len(tools)} tools via Streamable HTTP")
            return [{"name": tool.name, "description": getattr(tool, 'description', '')} for tool in tools]
            
    except Exception as e:
        logger.error(f"❌ Failed to get tools from {url} via Streamable HTTP: {str(e)}")
        return []

async def call_mcp_tool(tool_name: str, arguments: Dict[str, Any], server_url: str = None) -> Dict[str, Any]:
    """
    Call any MCP tool with given arguments using Streamable HTTP transport
    Returns standardized response format
    """
    url = server_url or MCP_SERVER_URL
    
    try:
        # FastMCP auto-detects Streamable HTTP transport from http/https URLs
        logger.info(f"🔧 Calling tool '{tool_name}' via Streamable HTTP: {url}")
        async with Client(url) as client:
            result = await client.call_tool(tool_name, arguments)
            
            # Extract content from result
            content = []
            for item in result:
                if hasattr(item, 'text'):
                    content.append(item.text)
                elif hasattr(item, 'content'):
                    content.append(item.content)
                else:
                    content.append(str(item))
            
            logger.info(f"✅ Tool '{tool_name}' executed successfully via Streamable HTTP")
            return {
                "success": True,
                "tool": tool_name,
                "result": content,
                "arguments": arguments,
                "transport": "streamable-http"
            }
            
    except Exception as e:
        logger.error(f"❌ Tool '{tool_name}' failed via Streamable HTTP: {str(e)}")
        return {
            "success": False,
            "tool": tool_name,
            "error": str(e),
            "arguments": arguments,
            "transport": "streamable-http"
        }

async def analyze_query_for_tools(query: str, available_tools: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze user query to determine which tools might be useful
    This is where you can add intelligence to automatically select tools
    """
    query_lower = query.lower()
    suggested_tools = []
    
    # Simple keyword matching - you can make this more sophisticated
    for tool in available_tools:
        tool_name = tool["name"].lower()
        tool_desc = tool.get("description", "").lower()
        
        # Check if query keywords match tool name or description
        if any(word in tool_name or word in tool_desc for word in query_lower.split()):
            suggested_tools.append(tool)
    
    return {
        "suggested_tools": suggested_tools,
        "should_use_mcp": len(suggested_tools) > 0,
        "analysis": f"Found {len(suggested_tools)} relevant tools for query"
    }

async def get_mcp_tools_context(query: str, server_url: str = None) -> Dict[str, Any]:
    """
    Main function to get MCP context for a query
    This is called by the Lambda function
    """
    url = server_url or MCP_SERVER_URL
    
    try:
        # Get available tools
        available_tools = await get_available_tools(url)
        
        if not available_tools:
            return {
                "context": "MCP tools are not available at this time.",
                "used_mcp": False,
                "results": {"error": "No tools available"}
            }
        
        # Analyze query to see if we should use MCP tools
        analysis = await analyze_query_for_tools(query, available_tools)
        
        if not analysis["should_use_mcp"]:
            # No relevant tools found, but still provide context about available tools
            tools_list = [tool["name"] for tool in available_tools]
            context = f"I have access to {len(available_tools)} MCP tools: {', '.join(tools_list)}. "
            context += "Let me know if you need help with any specific tasks that these tools might assist with."
            
            return {
                "context": context,
                "used_mcp": False,
                "results": {
                    "available_tools": tools_list,
                    "analysis": analysis["analysis"]
                }
            }
        
        # Build context with available tools and suggestions
        tools_list = [tool["name"] for tool in available_tools]
        suggested_tools = [tool["name"] for tool in analysis["suggested_tools"]]
        
        context = f"🔧 MCP TOOLS AVAILABLE (Streamable HTTP)\n\n"
        context += f"Connected to MCP server: {url}\n"
        context += f"Transport: Streamable HTTP (single endpoint)\n"
        context += f"Available tools: {len(available_tools)} | Suggested: {len(suggested_tools)}\n\n"
        context += f"All Tools: {', '.join(tools_list)}\n"
        if suggested_tools:
            context += f"Suggested for your query: {', '.join(suggested_tools)}\n\n"
        context += f"I can use these tools to help answer your question. What specific information do you need?"
        
        return {
            "context": context,
            "used_mcp": True,
            "results": {
                "available_tools": tools_list,
                "suggested_tools": suggested_tools,
                "analysis": analysis["analysis"],
                "server_url": url,
                "transport": "streamable-http"
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting MCP context: {str(e)}")
        return {
            "context": f"MCP tools encountered an error: {str(e)}",
            "used_mcp": False,
            "results": {"error": str(e)}
        }

async def execute_tool_by_name(tool_name: str, arguments: Dict[str, Any] = None, server_url: str = None) -> Dict[str, Any]:
    """
    Execute a specific tool by name
    Useful for when the AI decides it needs to call a specific tool
    """
    if arguments is None:
        arguments = {}
    
    return await call_mcp_tool(tool_name, arguments, server_url)

# Simple interface functions for common operations
async def list_available_tools(server_url: str = None) -> List[str]:
    """Get just the tool names"""
    tools = await get_available_tools(server_url)
    return [tool["name"] for tool in tools]

async def get_tool_info(tool_name: str, server_url: str = None) -> Dict[str, Any]:
    """Get detailed info about a specific tool"""
    tools = await get_available_tools(server_url)
    for tool in tools:
        if tool["name"] == tool_name:
            return tool
    return {"error": f"Tool '{tool_name}' not found"}

# Export main functions
__all__ = [
    'get_mcp_tools_context',
    'get_available_tools',
    'call_mcp_tool',
    'execute_tool_by_name',
    'list_available_tools',
    'get_tool_info'
]