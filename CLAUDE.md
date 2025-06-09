# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains an AWS Lambda function that implements a CDW Explore Advisory Bot using LangChain with Amazon Bedrock for the LLM (Claude) and DynamoDB for conversation history storage. The bot responds to user queries with professional technology consulting and advisory information. It also features FastMCP integration for healthcare data access via streamable HTTP transport.

## Architecture

- **Lambda Function**: Main application that processes user requests through AWS Lambda
- **LangChain + LangGraph**: Used for conversation management, modern memory handling with LangGraph StateGraph
- **Amazon Bedrock**: Provides Claude LLM models for generating responses
- **DynamoDB**: Stores conversation history for persistent memory between sessions
- **FastMCP**: Integration with Epic healthcare systems via streamable HTTP transport
- **Docker**: The application is containerized for deployment to AWS Lambda

## Code Structure

- **lambda_function.py**: Main Lambda handler with MCPConversationChain implementation
- **mcp_tools.py**: FastMCP streamable HTTP client for healthcare data integration
- **lambda_with_lex.py**: Alternative implementation that integrates with Amazon Lex
- **Dockerfile**: Container configuration for Lambda deployment
- **requirements.txt**: Python dependencies including LangChain, LangGraph, and FastMCP

## Development Commands

### Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Test the lambda function locally with a test payload:
   ```bash
   python -c "import lambda_function; import json; with open('test-payload.json') as f: event = json.load(f); print(lambda_function.lambda_handler(event, None))"
   ```

3. Test the lambda_with_lex implementation:
   ```bash
   python -c "import lambda_with_lex; import json; with open('test-payload.json') as f: event = json.load(f); print(lambda_with_lex.lambda_handler(event, None))"
   ```

### Building and Deployment

1. Build and deploy the Docker container to AWS Lambda:
   ```bash
   ./build_and_deploy.sh
   ```
   
   This script:
   - Creates an ECR repository (if not exists)
   - Builds the Docker image for Linux/AMD64
   - Tags and pushes the image to ECR
   - Updates or creates the Lambda function with proper role and memory settings

### Claude Code Execution

Run Claude Code with Amazon Bedrock:
```bash
./start_claude.sh
```

This script:
- Configures AWS credentials
- Sets up Bedrock integration with Claude model (currently Claude 3.7 Sonnet)
- Launches Claude Code for development

## Testing

Test the integration with different healthcare-related queries:

1. General query (should NOT use MCP):
   ```json
   {
     "body": {
       "message": "What are the latest trends in cloud computing?",
       "session_id": "test-session-1"
     }
   }
   ```

2. Healthcare query (should use MCP):
   ```json
   {
     "body": {
       "message": "Can you help me look up patient information in Epic?",
       "session_id": "test-session-2"
     }
   }
   ```

3. Direct patient lookup (MCP tools active):
   ```json
   {
     "body": {
       "message": "Find patient with ID e0w0LEDCYtfckT6N.CkJKCw3",
       "session_id": "test-session-3"
     }
   }
   ```

## Environment Configuration

The lambda function expects:
1. Access to Amazon Bedrock with Claude model
2. A DynamoDB table named "cdw-explore-advbot-chat-history"
3. MCP server configuration via environment variables:
   - `MCP_SERVER_URL`: URL of the Epic MCP server (default: https://epicmcp.cdwaws.com/mcp/)
   - `MCP_TIMEOUT_SECONDS`: Timeout for MCP requests (default: 15)
4. Appropriate AWS permissions via the "cdw-explore-advbot-lambda-role" IAM role

## Troubleshooting

1. If MCP tools are not available:
   - Check `MCP_SERVER_URL` environment variable
   - Verify server is accessible from Lambda VPC
   - Check Lambda memory settings (minimum 768MB recommended)

2. If tool execution times out:
   - Increase Lambda timeout
   - Check Epic server response times
   - Verify network connectivity

3. For Lambda deployment issues:
   - Ensure Docker is running with Linux/AMD64 platform support
   - Verify AWS credentials are properly configured
   - Check ECR repository permissions