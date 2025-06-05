"""
MCP Tools Integration Module
Handles FastMCP client connections via streamable HTTP transport for CDW Explore AdvBot
Optimized for Epic FHIR MCP server integration
"""
import json
import logging
from typing import Dict, List, Any, Optional
from fastmcp import Client
import os
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class MCPToolsManager:
    """
    Manages FastMCP client connections via streamable HTTP transport
    Provides a clean interface for the Lambda function to use MCP tools
    Optimized for Epic FHIR server integration
    """
    
    def __init__(self):
        self.server_url = os.environ.get('EPIC_MCP_SERVER_URL', 'https://epicmcp.cdwaws.com/mcp/')
        self.available_tools = []
        self.client = None
        self._tools_cache = {}
        
        # Streamable HTTP configuration
        self.timeout = int(os.environ.get('MCP_TIMEOUT_SECONDS', '30'))
        self.max_retries = int(os.environ.get('MCP_MAX_RETRIES', '3'))
        
        # Validate URL format for streamable HTTP
        if not self.server_url.startswith(('http://', 'https://')):
            raise ValueError(f"MCP server URL must use HTTP/HTTPS for streamable transport: {self.server_url}")
        
        logger.info(f"🌐 MCP Tools Manager initialized for streamable HTTP transport: {self.server_url}")
        
    async def initialize(self) -> bool:
        """Initialize MCP client and cache available tools"""
        try:
            # Test connection and get available tools using streamable HTTP transport
            # FastMCP Client automatically detects transport type from URL
            async with Client(self.server_url) as client:
                tools = await client.list_tools()
                self.available_tools = [tool.name for tool in tools]
                logger.info(f"✅ MCP initialized with {len(self.available_tools)} tools via streamable HTTP: {', '.join(self.available_tools)}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize MCP client via streamable HTTP: {str(e)}")
            return False
    
    def extract_content(self, response) -> Any:
        """Extract actual content from FastMCP response"""
        if hasattr(response, '__iter__') and len(response) > 0:
            first_content = response[0]
            if hasattr(first_content, 'text'):
                try:
                    return json.loads(first_content.text)
                except:
                    return first_content.text
        return response
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call an MCP tool using streamable HTTP transport with retry logic
        """
        try:
            # Use streamable HTTP transport via FastMCP Client with timeout
            timeout = httpx.Timeout(self.timeout)
            
            async with Client(self.server_url) as client:
                # Set timeout for the HTTP request
                response = await client.call_tool(tool_name, arguments)
                result = self.extract_content(response)
                
                logger.info(f"🔧 MCP tool '{tool_name}' executed successfully via streamable HTTP")
                
                return {
                    "success": True,
                    "tool": tool_name,
                    "result": result,
                    "arguments": arguments,
                    "transport": "streamable_http",
                    "server_url": self.server_url.split('//')[1].split('/')[0]  # Domain only for logging
                }
                
        except httpx.TimeoutException as e:
            logger.error(f"⏰ MCP tool '{tool_name}' timed out after {self.timeout}s: {str(e)}")
            return {
                "success": False,
                "tool": tool_name,
                "error": f"Request timed out after {self.timeout} seconds",
                "error_type": "timeout",
                "arguments": arguments,
                "transport": "streamable_http"
            }
        except httpx.ConnectError as e:
            logger.error(f"🔌 MCP connection failed for '{tool_name}': {str(e)}")
            return {
                "success": False,
                "tool": tool_name,
                "error": f"Failed to connect to MCP server: {str(e)}",
                "error_type": "connection",
                "arguments": arguments,
                "transport": "streamable_http"
            }
        except Exception as e:
            logger.error(f"❌ MCP tool '{tool_name}' failed via streamable HTTP: {str(e)}")
            return {
                "success": False,
                "tool": tool_name,
                "error": str(e),
                "error_type": "general",
                "arguments": arguments,
                "transport": "streamable_http"
            }
    
    async def get_patient_info(self, patient_id: str) -> Dict[str, Any]:
        """Get patient information by ID"""
        if "get_patient_by_id" not in self.available_tools:
            return {"success": False, "error": "Patient lookup tool not available"}
        
        return await self.call_tool("get_patient_by_id", {"patient_id": patient_id})
    
    async def match_patient(self, family_name: str, given_name: str, birth_date: str, 
                           phone_number: Optional[str] = None, email: Optional[str] = None) -> Dict[str, Any]:
        """Match patient using demographic information"""
        if "match_patient" not in self.available_tools:
            return {"success": False, "error": "Patient matching tool not available"}
        
        args = {
            "family_name": family_name,
            "given_name": given_name,
            "birth_date": birth_date
        }
        
        if phone_number:
            args["phone_number"] = phone_number
        if email:
            args["email"] = email
            
        return await self.call_tool("match_patient", args)
    
    async def search_patients(self, family_name: Optional[str] = None, 
                             given_name: Optional[str] = None, 
                             max_results: int = 5) -> Dict[str, Any]:
        """Search for patients"""
        if "search_patients" not in self.available_tools:
            return {"success": False, "error": "Patient search tool not available"}
        
        args = {"max_results": max_results}
        if family_name:
            args["family_name"] = family_name
        if given_name:
            args["given_name"] = given_name
            
        return await self.call_tool("search_patients", args)
    
    async def get_patient_appointments(self, patient_id: str, days_ahead: int = 30, 
                                     include_past: bool = False) -> Dict[str, Any]:
        """Get patient appointments"""
        if "get_patient_appointments" not in self.available_tools:
            return {"success": False, "error": "Appointment tool not available"}
        
        args = {
            "patient_id": patient_id,
            "days_ahead": days_ahead,
            "include_past": include_past
        }
        
        return await self.call_tool("get_patient_appointments", args)
    
    async def get_patient_complete_profile(self, patient_id: str, 
                                          include_appointments: bool = True,
                                          include_medications: bool = True) -> Dict[str, Any]:
        """Get comprehensive patient profile"""
        if "get_patient_complete_profile" not in self.available_tools:
            return {"success": False, "error": "Complete profile tool not available"}
        
        args = {
            "patient_id": patient_id,
            "include_appointments": include_appointments,
            "include_medications": include_medications,
            "appointment_days_ahead": 365,
            "appointment_days_back": 365
        }
        
        return await self.call_tool("get_patient_complete_profile", args)
    
    async def get_patient_medications_detailed(self, patient_id: str, 
                                              max_medications: int = 10) -> Dict[str, Any]:
        """Get detailed patient medications"""
        if "get_patient_medications_detailed" not in self.available_tools:
            return {"success": False, "error": "Medication details tool not available"}
        
        args = {
            "patient_id": patient_id,
            "include_medication_details": True,
            "max_medications": max_medications
        }
        
        return await self.call_tool("get_patient_medications_detailed", args)
    
    def get_available_tools(self) -> List[str]:
        """Get list of available MCP tools"""
        return self.available_tools.copy()
    
    def has_tool(self, tool_name: str) -> bool:
        """Check if a specific tool is available"""
        return tool_name in self.available_tools


class MCPAssistant:
    """
    High-level assistant that uses MCP tools to help with healthcare queries
    This provides natural language interface to MCP functionality
    """
    
    def __init__(self):
        self.mcp_manager = MCPToolsManager()
        self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the MCP assistant"""
        self.initialized = await self.mcp_manager.initialize()
        return self.initialized
    
    async def process_healthcare_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process healthcare-related queries using available MCP tools
        Returns structured response with tool results and natural language summary
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "MCP tools not available",
                "response": "I'm unable to access healthcare data systems at the moment. Please try again later."
            }
        
        query_lower = query.lower()
        results = []
        response_text = ""
        
        try:
            # Patient lookup patterns
            if "patient" in query_lower and any(word in query_lower for word in ["find", "search", "lookup", "get"]):
                
                # Extract patient ID if mentioned
                if "patient id" in query_lower or "id:" in query_lower:
                    # This is a simple pattern - you might want more sophisticated extraction
                    words = query.split()
                    for i, word in enumerate(words):
                        if word.lower() in ["id:", "id", "patient"] and i + 1 < len(words):
                            potential_id = words[i + 1].strip(".,!?")
                            if len(potential_id) > 10:  # Epic patient IDs are long
                                result = await self.mcp_manager.get_patient_info(potential_id)
                                results.append(result)
                                if result["success"]:
                                    response_text = "✅ Found patient information using the provided ID."
                                else:
                                    response_text = "❌ Could not find patient with that ID."
                                break
                
                # Extract name-based search
                elif any(name_word in query_lower for name_word in ["name", "named", "called"]):
                    response_text = "🔍 To search for patients by name, I would need specific first and last name. " \
                                  "For privacy and accuracy, please provide a patient ID if you have one."
                
                else:
                    response_text = "🔍 I can help you find patient information. Please provide either:\n" \
                                  "• A patient ID for direct lookup\n" \
                                  "• Specific first and last name for search"
            
            # Appointment queries
            elif "appointment" in query_lower:
                response_text = "📅 I can help with appointment information. Please provide a patient ID to see their appointments."
            
            # Medication queries  
            elif "medication" in query_lower or "prescription" in query_lower:
                response_text = "💊 I can provide detailed medication information. Please provide a patient ID to see their medications."
            
            # General healthcare info
            elif any(word in query_lower for word in ["epic", "fhir", "healthcare", "medical"]):
                tools_list = ", ".join(self.mcp_manager.get_available_tools())
                response_text = f"🏥 I have access to Epic FHIR healthcare data through these tools:\n{tools_list}\n\n" \
                               "I can help you look up patient information, appointments, medications, and more. " \
                               "What specific information do you need?"
            
            else:
                response_text = "I can help with healthcare data queries including patient lookup, appointments, " \
                               "and medications. What specific information are you looking for?"
            
            return {
                "success": True,
                "response": response_text,
                "mcp_results": results,
                "available_tools": self.mcp_manager.get_available_tools()
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing healthcare query: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": "I encountered an error while processing your healthcare query. Please try again."
            }
    
    async def execute_direct_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a direct MCP tool call - useful for specific tool execution
        """
        if not self.initialized:
            return {"success": False, "error": "MCP tools not available"}
        
        return await self.mcp_manager.call_tool(tool_name, arguments)


# Global instance for Lambda efficiency
mcp_assistant = None

async def get_mcp_assistant() -> MCPAssistant:
    """Get initialized MCP assistant instance"""
    global mcp_assistant
    if mcp_assistant is None:
        mcp_assistant = MCPAssistant()
        await mcp_assistant.initialize()
    return mcp_assistant

async def process_mcp_query(query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Main entry point for processing MCP queries from Lambda
    """
    assistant = await get_mcp_assistant()
    return await assistant.process_healthcare_query(query, context)

async def call_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Direct MCP tool call interface for Lambda
    """
    assistant = await get_mcp_assistant()
    return await assistant.execute_direct_tool_call(tool_name, arguments)

def is_healthcare_query(query: str) -> bool:
    """
    Determine if a query is healthcare-related and should use MCP tools
    """
    healthcare_keywords = [
        "patient", "appointment", "medication", "prescription", "epic", "fhir", 
        "healthcare", "medical", "doctor", "provider", "clinic", "hospital"
    ]
    
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in healthcare_keywords)

# Export the main functions for Lambda use
__all__ = [
    'process_mcp_query',
    'call_mcp_tool', 
    'is_healthcare_query',
    'get_mcp_assistant',
    'MCPAssistant',
    'MCPToolsManager'
]