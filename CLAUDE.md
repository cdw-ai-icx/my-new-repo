# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains an AWS Lambda function that implements a CDW Explore Advisory Bot using LangChain with Amazon Bedrock for the LLM (Claude) and DynamoDB for conversation history storage. The bot responds to user queries with professional technology consulting and advisory information.

## Architecture

- **Lambda Function**: The main application is an AWS Lambda function that processes user requests
- **LangChain**: Used for conversation chain management, memory handling, and LLM integration
- **Amazon Bedrock**: Provides the Claude LLM model for generating responses
- **DynamoDB**: Stores conversation history for persistent memory between sessions 
- **Docker**: The application is containerized for deployment to AWS Lambda

## Development Commands

### Local Development

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Test the lambda function locally with a test payload:
   ```
   python -c "import lambda_function; import json; with open('test-payload.json') as f: event = json.load(f); print(lambda_function.lambda_handler(event, None))"
   ```

### Building and Deployment

1. Build and deploy the Docker container to AWS Lambda:
   ```
   ./build_and_deploy.sh
   ```
   This script:
   - Creates an ECR repository (if not exists)
   - Builds the Docker image
   - Tags and pushes the image to ECR
   - Updates or creates the Lambda function

### Claude Code Execution

Run Claude Code with Amazon Bedrock:
```
./start_claude.sh
```

This script:
- Configures AWS credentials
- Sets up Bedrock integration with Claude model
- Launches Claude Code

## Key Files

- **lambda_function.py**: Contains the main Lambda handler function and all bot logic
- **requirements.txt**: Lists Python dependencies
- **Dockerfile**: Defines the container for Lambda deployment
- **build_and_deploy.sh**: Script for building and deploying to AWS
- **start_claude.sh**: Script for launching Claude Code with Bedrock
- **test-payload.json**: Sample payload for testing the Lambda function

## Environment Configuration

The lambda function expects:
1. Access to Amazon Bedrock with Claude model
2. A DynamoDB table named "cdw-explore-advbot-chat-history"
3. Appropriate AWS permissions via the "cdw-explore-advbot-lambda-role" IAM role