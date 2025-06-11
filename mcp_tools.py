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
    Get all available tools from the MCP server using FastMCP client
    Returns list of tool objects with name, description, etc.
    """
    url = server_url or MCP_SERVER_URL
    
    try:
        logger.info(f"🌐 Connecting to MCP server: {url}")
        
        # Use FastMCP Client with proper async context
        async with Client(url) as client:
            # Get tools using FastMCP list_tools method
            tools = await client.list_tools()
            logger.info(f"✅ Retrieved {len(tools)} tools from MCP server")
            
            # Convert to detailed format for LLM understanding
            tool_list = []
            for tool in tools:
                # Extract comprehensive tool information
                tool_dict = {
                    "name": tool.name,
                    "description": getattr(tool, 'description', f"Tool: {tool.name}"),
                    "input_schema": getattr(tool, 'inputSchema', {})
                }
                
                # Enhance description with parameter information for LLM
                schema = tool_dict["input_schema"]
                if schema and "properties" in schema:
                    param_info = []
                    for param_name, param_def in schema["properties"].items():
                        param_type = param_def.get("type", "string")
                        param_desc = param_def.get("description", "")
                        param_info.append(f"{param_name} ({param_type}): {param_desc}")
                    
                    if param_info:
                        tool_dict["description"] += f" Parameters: {'; '.join(param_info)}"
                
                tool_list.append(tool_dict)
                logger.info(f"📋 Tool: {tool.name}")
                logger.info(f"    Description: {tool_dict['description']}")
                if schema:
                    logger.info(f"    Schema: {schema}")
            
            return tool_list
            
    except Exception as e:
        logger.error(f"❌ Failed to get tools from {url}: {str(e)}")
        return []

async def call_mcp_tool(tool_name: str, arguments: Dict[str, Any], server_url: str = None) -> Dict[str, Any]:
    """
    Call any MCP tool with given arguments using FastMCP client
    Returns standardized response format
    """
    url = server_url or MCP_SERVER_URL
    
    try:
        # Fix validation for epic_headers parameter
        if 'epic_headers' in arguments and isinstance(arguments['epic_headers'], str):
            if arguments['epic_headers'] == '' or arguments['epic_headers'] == '{}':
                arguments['epic_headers'] = {}
        
        logger.info(f"🔧 Calling tool '{tool_name}' with args: {arguments}")
        
        # Use FastMCP Client with proper async context
        async with Client(url) as client:
            # Call tool using FastMCP call_tool method
            result = await client.call_tool(tool_name, arguments)
            
            # Extract content from FastMCP result
            content = []
            for item in result:
                if hasattr(item, 'text'):
                    content.append(item.text)
                elif hasattr(item, 'content'):
                    content.append(item.content)
                else:
                    content.append(str(item))
            
            logger.info(f"✅ Tool '{tool_name}' executed successfully. Result: {content[:100]}...")
            return {
                "success": True,
                "tool": tool_name,
                "result": content,
                "arguments": arguments
            }
            
    except Exception as e:
        logger.error(f"❌ Tool '{tool_name}' failed: {str(e)}")
        return {
            "success": False,
            "tool": tool_name,
            "error": str(e),
            "arguments": arguments
        }

async def analyze_query_for_tools(query: str, available_tools: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Simple analysis - let the LLM decide when to use tools based on tool schemas and user intent
    No hardcoded patterns or keyword detection
    """
    # Always make tools available - let LLM decide based on user context and tool capabilities
    should_use_mcp = len(available_tools) > 0
    
    logger.info(f"📋 Making {len(available_tools)} MCP tools available to LLM for intelligent selection")
    
    return {
        "suggested_tools": available_tools,  # Provide all tools, let LLM choose
        "should_use_mcp": should_use_mcp,
        "analysis": f"Available tools: {len(available_tools)} - LLM will decide usage based on user intent"
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
        
        # Analyze query - simplified approach
        analysis = await analyze_query_for_tools(query, available_tools)
        
        if not analysis["should_use_mcp"]:
            # No tools available
            return {
                "context": "MCP tools are not available right now.",
                "used_mcp": False,
                "results": {"error": "No tools available"}
            }
        
        # Build simple, clear context for LLM
        tools_list = [tool["name"] for tool in available_tools]
        
        # Create tool descriptions for LLM understanding
        tool_descriptions = []
        for tool in available_tools:
            tool_descriptions.append(f"- {tool['name']}: {tool['description']}")
        
        context = f"🔧 MCP TOOLS AVAILABLE\n\n"
        context += f"Connected to MCP server: {url}\n"
        context += f"Available tools ({len(available_tools)}):\n"
        context += "\n".join(tool_descriptions)
        context += f"\n\nThese tools are ready to use. Select appropriate tools based on the user's request and the tool parameters shown above."
        
        return {
            "context": context,
            "used_mcp": True,
            "results": {
                "available_tools": tools_list,
                "analysis": analysis["analysis"],
                "server_url": url
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