#!/bin/bash
# Run the bot validation script with proper AWS configuration

# Set AWS credentials
export AWS_PROFILE=cdw-demo
export AWS_REGION=us-east-1

# Ensure Python environment is activated
if [ -d "venv" ]; then
  echo "Activating virtual environment..."
  source venv/bin/activate
fi

# Ensure dependencies are installed
echo "Checking for dependencies..."
pip install -q -r requirements.txt

# Run validation script
echo "Starting bot validation..."

# Check for arguments
if [ "$1" == "--health-check" ]; then
  echo "Running health check..."
  python bot-validation.py --mode health-check
elif [ "$1" == "--batch" ]; then
  echo "Running batch tests..."
  python bot-validation.py --mode batch --test-file test-cases.json --output batch-results-$(date +%Y%m%d-%H%M%S).json
else
  echo "Starting interactive session..."
  python bot-validation.py --mode interactive
fi

# Check exit status
exit_status=$?
if [ $exit_status -ne 0 ]; then
  echo "Bot validation failed with status: $exit_status"
  exit $exit_status
fi

echo "Bot validation completed successfully"