# Bot Validation Tool

This tool provides a standalone environment for testing and validating the CDW Explore Advisory Bot without Lambda or Lex dependencies. It connects directly to AWS Bedrock for LLM capabilities and the MCP server for healthcare data integration.

## Features

- **Standalone Testing**: Test bot functionality without deploying to AWS Lambda
- **Interactive Mode**: Chat with the bot through a command-line interface
- **Batch Testing**: Run predefined test cases and analyze results
- **Health Checks**: Validate connections to AWS Bedrock and MCP server
- **Multi-step Interactions**: Test conversation flows across multiple turns
- **MCP Tool Integration**: Test Epic healthcare data access via FastMCP
- **Comprehensive Logging**: Detailed logs for debugging and analysis

## Requirements

- Python 3.9+
- AWS credentials with Bedrock access
- MCP server connection (configured in `mcp_tools.py`)
- Dependencies from `requirements.txt`

## Usage

### Interactive Mode

To have a conversation with the bot:

```bash
./run-bot-validation.sh
```

This starts an interactive session where you can chat with the bot directly.

### Health Check

To validate AWS Bedrock and MCP server connectivity:

```bash
./run-bot-validation.sh --health-check
```

This performs connectivity checks and returns status information.

### Batch Testing

To run predefined test cases:

```bash
./run-bot-validation.sh --batch
```

This runs through all test cases in `test-cases.json` and saves results to a JSON file.

## Test Cases

The included `test-cases.json` file contains various test scenarios:

1. General information queries (not using MCP)
2. Healthcare-related questions
3. Direct patient lookups
4. Appointment information requests
5. Error handling with invalid requests
6. Multi-step interaction flows
7. Tool-chaining scenarios

## Advanced Usage

For more control, you can run the Python script directly:

```bash
python bot-validation.py --mode [interactive|batch|health-check] --test-file test-cases.json --output results.json
```

## Logging

Detailed logs are written to:
- Console output
- `bot_validation.log` file

## Troubleshooting

Common issues:
- AWS credentials: Ensure `cdw-demo` profile exists in your AWS config
- MCP connectivity: Check `MCP_SERVER_URL` in `mcp_tools.py`
- Dependencies: Ensure all packages in `requirements.txt` are installed