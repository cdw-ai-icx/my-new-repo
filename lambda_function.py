import json
import boto3
import os
import asyncio
from typing import Dict, Any, TypedDict, List
from langchain_aws import ChatBedrock
from langchain_community.chat_message_histories import DynamoDBChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Import LangGraph for modern conversation management (replaces deprecated memory)
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

# Import our MCP tools module
from mcp_tools import get_mcp_tools_context

# Define the conversation state for LangGraph (modern approach for memory management)
class ConversationState(TypedDict):
    messages: List[BaseMessage]
    current_input: str
    mcp_context: str

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

{mcp_context}

IMPORTANT PATIENT DATA SECURITY RULES (APPLICATION LEVEL - NOT MCP LEVEL):
- For ANY patient data request, you MUST validate identity first
- For general inquiries: Require full name, date of birth, AND either phone number OR email address
- For patient ID inquiries: Require patient ID, full name, AND date of birth for verification
- Never access patient data without proper identity validation
- Be friendly but firm about security requirements
- Tools that likely access patient data: search_patients, get_patient_*, match_patient, *_appointments, *_medication*

Use the conversation history to provide contextual, caring responses.

Conversation History:
{history}

Current Question: {input}

Respond as Medley - be friendly, clear, and concise. Always prioritize patient privacy and security. Use available MCP tools when appropriate and safe."""
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
        
        # Create chain and execute
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
        
        # Generate response
        response = chain.invoke({})
        
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


def lambda_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Enhanced LangChain Lambda function handler with FastMCP integration
    Production-ready with proper error handling and timeouts
    """
    try:
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