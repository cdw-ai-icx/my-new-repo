#!/bin/bash

# Use default AWS credentials or profile
export AWS_PROFILE=default
echo "Using $AWS_PROFILE for AWS credentials"

# Verify AWS credentials are accessible
echo "Checking AWS credentials..."
if ! aws sts get-caller-identity &>/dev/null; then
  echo "AWS credentials not found or invalid. Please check your AWS configuration."
  exit 1
fi
echo "AWS credentials are valid!"

# Configure for Bedrock with Claude 3.7 Sonnet
export CLAUDE_CODE_USE_BEDROCK=1
export ANTHROPIC_MODEL='us.anthropic.claude-sonnet-4-20250514-v1:0'
#export ANTHROPIC_MODEL='us.anthropic.claude-3-7-sonnet-20250219-v1:0'
# Launch Claude Code
echo "Starting Claude Code..."
claude