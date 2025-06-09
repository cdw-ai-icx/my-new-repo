import json
import boto3
import os
import logging
import asyncio
from typing import Dict, Any, TypedDict, List, Optional
from botocore.exceptions import ClientError, BotoCoreError
from langchain_aws import ChatBedrock
from langchain_community.chat_message_histories import DynamoDBChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Import LangGraph for modern conversation management (replaces deprecated memory)
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

# Import our MCP tools module
from mcp_tools import get_mcp_tools_context, get_available_tools, call_mcp_tool

# Set up structured logging
logger = logging.getLogger()
if logger.handlers:
    for handler in logger.handlers:
        logger.removeHandler(handler)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s", "function": "%(funcName)s"}'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Initialize AWS clients with error handling
def get_bedrock_client():
    """Get Bedrock client with error handling"""
    try:
        return boto3.client("bedrock-runtime")
    except Exception as e:
        logger.error(f"Failed to initialize Bedrock client: {str(e)}")
        raise

# Only initialize client when needed (not at import time)
bedrock_client = None

# Helper functions for Lex integration
def sanitize_text(text: str, max_length: int = 1000) -> str:
    """Sanitize and truncate text for Lex compatibility."""
    if not text or not isinstance(text, str):
        return "Hello"
    
    # Remove control characters and excessive whitespace
    sanitized = ' '.join(text.strip().split())
    
    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length-3] + "..."
    
    return sanitized or "Hello"

def create_error_response(session_attributes: Dict, message: str = "I'm having trouble right now. Please try again in a moment.") -> Dict:
    """Create a standardized error response for Lex."""
    return {
        "sessionState": {
            "dialogAction": {"type": "ElicitIntent"},
            "intent": {"name": "FallbackIntent", "state": "InProgress", "slots": {}},
            "sessionAttributes": session_attributes or {}
        },
        "messages": [{
            "contentType": "PlainText",
            "content": sanitize_text(message, 950)
        }]
    }

# Define the conversation state for LangGraph (modern approach for memory management)
class ConversationState(TypedDict):
    messages: List[BaseMessage]
    current_input: str
    mcp_context: str
    end_conversation: bool
    transfer_to_agent: bool

class MCPConversationChain:
    """
    Modern conversation chain using LangChain Expression Language (LCEL)
    and LangGraph for state management (replacing deprecated memory classes)
    """
    
    def __init__(self, llm, message_store=None, verbose=False):
        self.llm = llm
        self.verbose = verbose
        self.message_store = message_store  # DynamoDBChatMessageHistory instance
        self.max_messages = 10  # Keep history manageable
        
        # Initialize message history from store or create empty
        self.messages = []
        if message_store and hasattr(message_store, 'messages'):
            initial_messages = message_store.messages[-self.max_messages:] if message_store.messages else []
            self.messages = list(initial_messages)
        
        # Create memory saver for persistence
        self.memory = MemorySaver()
        
        # Modern prompt template for Medley - CDW Healthcare Bot
        self.prompt_template = PromptTemplate(
            input_variables=["history", "input", "mcp_context"],
            template="""You are Medley, CDW Healthcare's friendly and helpful patient assistant. You help patients with appointments, medications, finding the right healthcare team members, and connecting them with the care they need.

You have access to Epic healthcare systems through MCP tools in a secure test environment with real test patient data.

{mcp_context}

WHEN AND HOW TO USE MCP TOOLS:
- You SHOULD use MCP tools when patients ask for their information (appointments, medications, records, etc.)
- Patient ID provided: Use tools like get_patient_by_id, search_patients, get_patient_appointments
- Patient asking about appointments: Use get_patient_appointments or search_appointments
- Patient asking about medications: Use get_patient_medications
- Patient asking to find their record: Use search_patients with their details
- These are test Epic systems - you CAN and SHOULD access patient data when requested

SECURITY VALIDATION (for actual data access):
- For patient ID requests: Use the patient ID they provide directly with tools
- For general searches: Ask for name and date of birth if needed for the specific tool
- Always use the tools when patient data is requested - don't refuse access
- DO NOT MAKE UP DATA - always use actual tool results

Use the conversation history to provide contextual, caring responses.

Conversation History:
{history}

Current Question: {input}

Respond as Medley - be friendly, clear, and concise. When patients ask for their healthcare information, USE THE AVAILABLE MCP TOOLS to get their actual data from Epic systems."""
        )
        
        # Define the graph for conversation flow
        self._create_graph()
    
    def _create_graph(self):
        """Create LangGraph for state management"""
        # Initialize a state graph with our typed state
        self.graph = StateGraph(ConversationState)
        
        # Add node for generating response
        self.graph.add_node("generate", self._generate_response)
        
        # Set the entry point
        self.graph.set_entry_point("generate")
        
        # Compile the graph
        self.graph = self.graph.compile()
    
    def _generate_response(self, state: ConversationState) -> ConversationState:
        """Generate response based on current state"""
        history_str = self._format_history(state['messages'])
        
        # Handle MCP tool execution with proper tool calling loop
        response = self._handle_tool_execution(history_str, state)
        
        # Update message history
        new_messages = list(state['messages'])  # Copy existing messages
        new_messages.append(HumanMessage(content=state['current_input']))
        new_messages.append(AIMessage(content=response))
        
        # Keep only last N messages
        if len(new_messages) > self.max_messages * 2:  # Human + AI messages
            new_messages = new_messages[-(self.max_messages * 2):]
        
        # Update DynamoDB storage if available
        if self.message_store:
            try:
                self.message_store.add_message(HumanMessage(content=state['current_input']))
                self.message_store.add_message(AIMessage(content=response))
            except Exception as e:
                print(f"Warning: Could not save to DynamoDB: {str(e)}")
        
        # Update internal message list
        self.messages = new_messages
        
        # Return updated state
        return {
            "messages": new_messages,
            "current_input": state['current_input'],
            "mcp_context": state['mcp_context']
        }

    def _handle_tool_execution(self, history_str: str, state: ConversationState) -> str:
        """Handle tool execution with proper tool calling loop"""
        import asyncio
        from langchain_core.messages import ToolMessage
        
        # Check if MCP tools are available
        if not (state['mcp_context'] and "MCP TOOLS AVAILABLE" in state['mcp_context']):
            # No MCP tools available, use regular chain
            chain = (
                {
                    "history": lambda _: history_str,
                    "input": lambda _: state['current_input'],
                    "mcp_context": lambda _: state['mcp_context']
                }
                | self.prompt_template
                | self.llm
                | StrOutputParser()
            )
            return chain.invoke({})
        
        # Get available tools and bind them to the LLM
        try:
            loop = asyncio.get_event_loop()
            available_tools = loop.run_until_complete(get_available_tools())
            
            if not available_tools:
                # Fallback to regular chain if no tools available
                chain = (
                    {
                        "history": lambda _: history_str,
                        "input": lambda _: state['current_input'],
                        "mcp_context": lambda _: state['mcp_context']
                    }
                    | self.prompt_template
                    | self.llm
                    | StrOutputParser()
                )
                return chain.invoke({})
            
            # Create tool functions for LangChain using proper schema
            from langchain_core.tools import StructuredTool
            from pydantic import BaseModel, Field
            
            # Create LangChain tools from MCP tools with proper schemas
            langchain_tools = []
            for tool in available_tools:
                # Create a dynamic schema for each tool
                def create_tool_schema(tool_name: str):
                    # Create a flexible schema that accepts any arguments
                    class DynamicToolInput(BaseModel):
                        # This will allow any field names
                        class Config:
                            extra = "allow"
                    
                    return DynamicToolInput
                
                def create_mcp_tool_wrapper(tool_name: str):
                    def mcp_tool_func(**kwargs):
                        loop = asyncio.get_event_loop()
                        # Pass arguments directly without wrapping
                        result = loop.run_until_complete(call_mcp_tool(tool_name, kwargs))
                        if result.get("success"):
                            return "\n".join(result.get("result", []))
                        else:
                            return f"Error: {result.get('error', 'Unknown error')}"
                    return mcp_tool_func
                
                tool_func = create_mcp_tool_wrapper(tool["name"])
                tool_schema = create_tool_schema(tool["name"])
                
                lc_tool = StructuredTool(
                    name=tool["name"],
                    description=tool.get("description", f"MCP tool: {tool['name']}"),
                    func=tool_func,
                    args_schema=tool_schema
                )
                langchain_tools.append(lc_tool)
            
            # Bind tools to LLM
            llm_with_tools = self.llm.bind_tools(langchain_tools)
            logger.info(f"Bound {len(langchain_tools)} MCP tools to LLM")
            
            # Initial prompt with tools
            messages = [
                {"role": "system", "content": self.prompt_template.format(
                    history=history_str,
                    input=state['current_input'],
                    mcp_context=state['mcp_context']
                )},
                {"role": "user", "content": state['current_input']}
            ]
            
            # Tool execution loop - reduced iterations to prevent timeout
            max_iterations = 2
            for i in range(max_iterations):
                # Get response from LLM
                response = llm_with_tools.invoke(messages)
                
                # Check if LLM wants to use tools
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    logger.info(f"LLM requested {len(response.tool_calls)} tool calls")
                    messages.append(response)
                    
                    # Execute each tool call
                    for tool_call in response.tool_calls:
                        tool_name = tool_call['name']
                        tool_args = tool_call.get('args', {})
                        tool_id = tool_call.get('id', f"call_{i}_{tool_name}")
                        
                        logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
                        
                        # Find and execute the tool directly via MCP
                        try:
                            # Check if arguments are wrapped with 'kwargs' and unwrap if needed
                            if 'kwargs' in tool_args and len(tool_args) == 1:
                                # Unwrap the arguments
                                actual_args = tool_args['kwargs']
                                logger.info(f"Unwrapping kwargs for tool {tool_name}: {actual_args}")
                            else:
                                actual_args = tool_args
                            
                            # Use the MCP call directly
                            result = asyncio.get_event_loop().run_until_complete(call_mcp_tool(tool_name, actual_args))
                            if result.get("success"):
                                tool_result = "\n".join(result.get("result", []))
                            else:
                                tool_result = f"MCP Error: {result.get('error', 'Unknown error')}"
                        except Exception as e:
                            tool_result = f"Error executing tool: {str(e)}"
                        
                        # Add tool result to messages
                        messages.append(ToolMessage(
                            content=str(tool_result),
                            tool_call_id=tool_id
                        ))
                    
                    # Continue the loop to get final response
                    continue
                else:
                    # No tool calls, return the response
                    return response.content if hasattr(response, 'content') else str(response)
            
            # If we hit max iterations, return the last response
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            logger.warning(f"Tool execution error: {str(e)}")
            # Fallback to regular chain
            chain = (
                {
                    "history": lambda _: history_str,
                    "input": lambda _: state['current_input'],
                    "mcp_context": lambda _: state['mcp_context']
                }
                | self.prompt_template
                | self.llm
                | StrOutputParser()
            )
            return chain.invoke({})
    
    def predict(self, input_text: str, mcp_context: str = "") -> str:
        """Modern prediction using LangGraph"""
        # Create initial state
        state = {
            "messages": self.messages,
            "current_input": input_text,
            "mcp_context": mcp_context
        }
        
        # Invoke the graph
        result = self.graph.invoke(state)
        
        # Return generated response
        return result["messages"][-1].content if result["messages"] else "I'm sorry, I couldn't generate a response."
    
    def _format_history(self, messages: List[BaseMessage]) -> str:
        """Format message history for the prompt"""
        if not messages:
            return "No previous conversation."
        
        formatted = []
        for msg in messages[-20:]:  # Last 20 messages to avoid token limits
            if isinstance(msg, HumanMessage):
                formatted.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):  
                formatted.append(f"Assistant: {msg.content}")
            else:
                # Fallback for different message types
                formatted.append(str(msg))
        
        return "\n".join(formatted)


async def get_mcp_context_async(user_message: str) -> Dict[str, Any]:
    """
    Async wrapper for MCP context retrieval with shorter timeout
    """
    try:
        # Reduced timeout for faster responses
        mcp_context_result = await asyncio.wait_for(
            get_mcp_tools_context(user_message),
            timeout=15.0  # Reduced from 30 to 15 seconds
        )
        return mcp_context_result
    except asyncio.TimeoutError:
        print("⏰ MCP context retrieval timed out")
        return {
            "context": "Note: MCP tools request timed out.",
            "used_mcp": False,
            "results": {"error": "timeout"}
        }
    except Exception as e:
        print(f"❌ MCP Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "context": "Note: MCP tools are temporarily unavailable.",
            "used_mcp": False,
            "results": {"error": str(e)}
        }


def build_messages(conversation_history: str, user_input: str) -> List[Dict]:
    """Build message list from conversation history and current input."""
    messages = []
    
    if conversation_history.strip():
        pairs = [p for p in conversation_history.split("User: ") if p.strip()]
        for pair in pairs:
            if "Assistant: " not in pair:
                continue
            user_msg, assistant_msg = pair.split("Assistant: ", 1)
            if user_msg.strip() and assistant_msg.strip():
                messages.extend([
                    {"role": "user", "content": [{"type": "text", "text": user_msg.strip()}]},
                    {"role": "assistant", "content": [{"type": "text", "text": assistant_msg.strip()}]}
                ])

    # Add current user input (sanitized)
    sanitized_input = sanitize_text(user_input)
    messages.append({
        "role": "user",
        "content": [{"type": "text", "text": sanitized_input}]
    })
    
    return messages

def detect_lex_event(event: Dict) -> bool:
    """Detect if this is a Lex event based on event structure."""
    # Key Lex fields that distinguish it from API Gateway events
    has_input_transcript = "inputTranscript" in event
    has_session_state = "sessionState" in event
    has_message_version = "messageVersion" in event
    has_invocation_source = "invocationSource" in event
    
    # A true Lex event should have inputTranscript AND sessionState
    is_lex = has_input_transcript and has_session_state
    
    logger.info(f"Lex detection: inputTranscript={has_input_transcript}, sessionState={has_session_state}, messageVersion={has_message_version}, invocationSource={has_invocation_source} -> is_lex={is_lex}")
    return is_lex

def handle_lex_event(event: Dict, context) -> Dict:
    """Handle Lex-specific event processing using existing LangChain infrastructure."""
    request_id = context.aws_request_id if context else "unknown"
    logger.info(f"Processing Lex request {request_id}")
    
    # Validate event structure
    if not isinstance(event, dict):
        logger.error("Invalid event format - not a dictionary")
        return create_error_response({})

    # Extract and validate event data
    session_state = event.get("sessionState", {})
    session_attributes = session_state.get("sessionAttributes", {})
    intent = session_state.get("intent", {"name": "FallbackIntent"})
    
    user_input = sanitize_text(event.get("inputTranscript", ""))
    conversation_history = session_attributes.get("conversation_history", "")
    
    logger.info(f"Processing Lex input: {user_input[:100]}..." if len(user_input) > 100 else f"Processing Lex input: {user_input}")
    
    try:
        # Use session ID from Lex event (at root level)
        session_id = event.get("sessionId", "lex-default-session")
        
        # Initialize Bedrock Chat Model with optimized settings for Lex
        llm = ChatBedrock(
            model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
            region_name=os.environ.get('AWS_REGION', 'us-east-1'),
            model_kwargs={
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 2000
            }
        )
        
        # Setup DynamoDB Chat History for Lex session
        message_history = DynamoDBChatMessageHistory(
            table_name="cdw-explore-advbot-chat-history",
            session_id=session_id,
            boto3_session=boto3.Session()
        )
        
        # Get MCP tools context with proper async handling
        logger.info(f"Getting MCP tools context for Lex query...")
        
        # Handle async MCP context retrieval
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, get_mcp_context_async(user_input))
                    mcp_context_result = future.result(timeout=35)
            else:
                mcp_context_result = loop.run_until_complete(get_mcp_context_async(user_input))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                mcp_context_result = loop.run_until_complete(get_mcp_context_async(user_input))
            finally:
                loop.close()
        
        # Extract MCP context
        mcp_context = mcp_context_result.get("context", "")
        mcp_used = mcp_context_result.get("used_mcp", False)
        
        logger.info(f"MCP context retrieved for Lex. Used MCP: {mcp_used}")
        
        # Create MCP-enabled conversation chain
        conversation_chain = MCPConversationChain(
            llm=llm,
            message_store=message_history,
            verbose=False
        )
        
        # Generate response with MCP context integration
        logger.info(f"Generating Lex response with LangChain...")
        response = conversation_chain.predict(
            input_text=user_input,
            mcp_context=mcp_context
        )
        
        logger.info(f"Lex response generated successfully")
        
        # Ensure response fits Lex limits (1000 char max)
        generated_response = sanitize_text(response, 950)
        logger.info(f"Generated Lex response length: {len(generated_response)}")
        
        # Update conversation history for Lex session attributes
        updated_history = f"{conversation_history}\nUser: {user_input}\nAssistant: {generated_response}"
        
        # Update session attributes
        updated_session_attributes = session_attributes.copy()
        updated_session_attributes["conversation_history"] = updated_history
        
        # Check for transfer or end markers
        if "**Transfer_to_Agent**" in generated_response or "**END_OF_CONVERSATION**" in generated_response:
            end_reason = "Transfer_to_Agent" if "**Transfer_to_Agent**" in generated_response else "END_OF_CONVERSATION"
            clean_response = generated_response.replace(f"**{end_reason}**", "").strip()
            logger.info(f"Lex conversation ending with reason: {end_reason}")
            
            return {
                "sessionState": {
                    "dialogAction": {"type": "Close"},
                    "intent": {
                        "name": intent.get("name", "FallbackIntent"),
                        "state": "Fulfilled",
                        "slots": intent.get("slots", {})
                    },
                    "sessionAttributes": updated_session_attributes
                },
                "messages": [{
                    "contentType": "PlainText",
                    "content": clean_response
                }]
            }
        
        # Continue Lex conversation
        return {
            "sessionState": {
                "dialogAction": {"type": "ElicitIntent"},
                "intent": {
                    "name": intent.get("name", "FallbackIntent"),
                    "state": "InProgress",
                    "slots": intent.get("slots", {})
                },
                "sessionAttributes": updated_session_attributes
            },
            "messages": [{
                "contentType": "PlainText",
                "content": generated_response
            }]
        }
        
    except Exception as error:
        logger.exception(f"Unexpected error in Lex handler: {str(error)}")
        return create_error_response(session_attributes, "Something unexpected happened. Please try again.")

def lambda_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Enhanced Lambda handler supporting both regular API events and Lex events
    with FastMCP integration and LangChain conversation management
    """
    try:
        logger.info(f"Received event keys: {list(event.keys())}")
        logger.info(f"Event has body: {'body' in event}, has inputTranscript: {'inputTranscript' in event}, has sessionState: {'sessionState' in event}")
        
        # Detect event type and route accordingly
        if detect_lex_event(event):
            logger.info("Detected Lex event - routing to Lex handler")
            return handle_lex_event(event, context)
        
        # Handle regular API events
        logger.info("Detected regular API event - using standard handler")
        
        # Parse input
        body = json.loads(event.get('body', '{}')) if isinstance(event.get('body'), str) else event.get('body', {})
        user_message = body.get('message', '')
        session_id = body.get('session_id', 'default-session')
        
        if not user_message:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'Message is required'})
            }
        
        print(f"🚀 Processing message: {user_message[:100]}...")
        
        # Initialize Bedrock Chat Model with optimized settings
        llm = ChatBedrock(
            model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
            region_name=os.environ.get('AWS_REGION', 'us-east-1'),
            model_kwargs={
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 2000  # Limit tokens for faster responses
            }
        )
        
        # Setup DynamoDB Chat History
        message_history = DynamoDBChatMessageHistory(
            table_name="cdw-explore-advbot-chat-history",
            session_id=session_id,
            boto3_session=boto3.Session()
        )
        
        # Using DynamoDB chat message history directly - LangGraph will handle window limiting
        # No need for WindowChatMessageHistory or other deprecated memory classes
        
        # Get MCP tools context with proper async handling
        print(f"🔧 Getting MCP tools context for query...")
        
        # FIXED: Better event loop handling for Lambda
        try:
            # Try to get existing event loop first
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, we need to handle this differently
                # This shouldn't happen in Lambda, but just in case
                print("⚠️ Event loop already running, using asyncio.create_task")
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, get_mcp_context_async(user_message))
                    mcp_context_result = future.result(timeout=35)  # 35s timeout
            else:
                # Normal case - run in the existing loop
                mcp_context_result = loop.run_until_complete(get_mcp_context_async(user_message))
        except RuntimeError:
            # No event loop exists, create one
            print("🔄 Creating new event loop")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                mcp_context_result = loop.run_until_complete(get_mcp_context_async(user_message))
            finally:
                loop.close()
        
        # Extract MCP context
        mcp_context = mcp_context_result.get("context", "")
        mcp_used = mcp_context_result.get("used_mcp", False)
        mcp_results = mcp_context_result.get("results", {})
        
        print(f"✅ MCP context retrieved. Used MCP: {mcp_used}")
        
        # Create our custom MCP-enabled conversation chain using LangGraph for memory
        conversation_chain = MCPConversationChain(
            llm=llm,
            message_store=message_history,  # Pass DynamoDB history directly
            verbose=False  # Set to False for production to reduce logs
        )
        
        # Get response with MCP context integration
        print(f"🚀 Generating response with LangChain...")
        response = conversation_chain.predict(
            input_text=user_message,
            mcp_context=mcp_context
        )
        
        print(f"✅ Response generated successfully")
        
        # Enhanced response with MCP metadata
        response_body = {
            'response': response,
            'session_id': session_id,
            'bot_name': 'Medley - CDW Healthcare Assistant',
            'model': 'us.anthropic.claude-sonnet-4-20250514-v1:0',
            'message_count': len(conversation_chain.messages),
            'status': 'success',
            'mcp_enabled': True,
            'used_mcp': mcp_used,
            'mcp_results': mcp_results,
            'timestamp': context.aws_request_id if context else 'local-test',
            'healthcare_focused': True
        }
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
                'X-MCP-Enabled': 'true',
                'X-Request-ID': context.aws_request_id if context else 'local-test'
            },
            'body': json.dumps(response_body)
        }
        
    except Exception as e:
        print(f"❌ Lambda Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'error': 'Internal server error',
                'details': str(e),
                'bot_name': 'Medley - CDW Healthcare Assistant',
                'mcp_enabled': True,
                'error_type': 'lambda_execution',
                'request_id': context.aws_request_id if context else 'local-test'
            })
        }