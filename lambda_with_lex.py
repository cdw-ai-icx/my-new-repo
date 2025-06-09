import json
import boto3
import logging
import os
from typing import Dict, List, Optional
from botocore.exceptions import ClientError, BotoCoreError

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

# Environment variables with validation
try:
    BEDROCK_MODEL_ID = os.environ["BEDROCK_MODEL_ID"]
    logger.info(f"Using Bedrock model: {BEDROCK_MODEL_ID}")
except KeyError:
    logger.error("BEDROCK_MODEL_ID environment variable not set")
    raise

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

# Simple system prompt
SYSTEM_PROMPT = """You are a friendly virtual assistant supporting customer interactions in a CDW Amazon Connect demo environment.

Keep your responses natural, friendly, and concise. Break conversations into small, easy steps and avoid long messages.

If the customer asks to speak to a live agent, confirm the transfer with "I'm transferring you now" and include:
**Transfer_to_Agent**

If the conversation is complete or the user says they're done, close politely and include:
**END_OF_CONVERSATION**

Conversation History: {conversation_history}
"""

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

def lambda_handler(event: Dict, context) -> Dict:
    """Main Lambda handler for Lex-Bedrock integration."""
    request_id = context.aws_request_id if context else "unknown"
    logger.info(f"Processing request {request_id}")
    
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
    
    logger.info(f"Processing input: {user_input[:100]}..." if len(user_input) > 100 else f"Processing input: {user_input}")
    
    try:
        # Format system prompt with conversation history
        formatted_prompt = SYSTEM_PROMPT.format(
            conversation_history=conversation_history
        )
        
        # Build messages for Bedrock
        messages = build_messages(conversation_history, user_input)
        system_prompts = [{"text": formatted_prompt}]
        inference_config = {"temperature": 0.7, "maxTokens": 2000}
        
        # Initialize client if needed
        global bedrock_client
        if bedrock_client is None:
            bedrock_client = get_bedrock_client()
            logger.info("Bedrock client initialized successfully")
        
        # Call Bedrock with specific error handling
        try:
            logger.info(f"Calling Bedrock with model {BEDROCK_MODEL_ID}")
            # Build request for Claude
            claude_request = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": inference_config.get("maxTokens", 2000),
                "temperature": inference_config.get("temperature", 0.7),
                "system": formatted_prompt,
                "messages": messages
            }
            
            response = bedrock_client.invoke_model(
                modelId=BEDROCK_MODEL_ID,
                body=json.dumps(claude_request),
                contentType="application/json"
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            logger.info("Bedrock call successful")
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            logger.error(f"Bedrock ClientError: {error_code} - {str(e)}")
            if error_code == 'ThrottlingException':
                return create_error_response(session_attributes, "The system is busy right now. Please try again in a moment.")
            elif error_code == 'ValidationException':
                return create_error_response(session_attributes, "I had trouble understanding your request. Could you rephrase it?")
            else:
                return create_error_response(session_attributes)
        except BotoCoreError as e:
            logger.error(f"Bedrock BotoCoreError: {str(e)}")
            return create_error_response(session_attributes, "I'm having connectivity issues. Please try again shortly.")
        
        # Extract and validate response text
        content_blocks = response_body.get("content", [])
        if not content_blocks:
            logger.warning("No content blocks in Bedrock response")
            return create_error_response(session_attributes, "I didn't generate a proper response. Please try again.")
        
        generated_response = " ".join(
            block.get("text", "") for block in content_blocks if block.get("type") == "text"
        ).strip()
        
        if not generated_response:
            logger.warning("Empty response from Bedrock")
            generated_response = "Sorry, I didn't catch that. Could you say that again?"
        
        # Ensure response fits Lex limits (1000 char max)
        generated_response = sanitize_text(generated_response, 950)
        logger.info(f"Generated response length: {len(generated_response)}")
        
        # Update conversation history
        updated_history = f"{conversation_history}\nUser: {user_input}\nAssistant: {generated_response}"
        
        # Update session attributes
        updated_session_attributes = session_attributes.copy()
        updated_session_attributes["conversation_history"] = updated_history
        
        # Check for transfer or end markers
        if "**Transfer_to_Agent**" in generated_response or "**END_OF_CONVERSATION**" in generated_response:
            end_reason = "Transfer_to_Agent" if "**Transfer_to_Agent**" in generated_response else "END_OF_CONVERSATION"
            clean_response = generated_response.replace(f"**{end_reason}**", "").strip()
            logger.info(f"Conversation ending with reason: {end_reason}")
            
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
        
        # Continue conversation
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
        logger.exception(f"Unexpected error in lambda_handler: {str(error)}")
        return create_error_response(session_attributes, "Something unexpected happened. Please try again.")