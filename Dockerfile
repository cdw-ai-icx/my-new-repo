
# Use the official AWS Lambda Python 3.13 base image
FROM public.ecr.aws/lambda/python:3.13

# Copy requirements file
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install build dependencies for langgraph (needed for building wheels)
RUN pip install --upgrade pip
RUN pip install wheel setuptools

# Install Python dependencies with better error reporting
RUN pip install -r requirements.txt --target ${LAMBDA_TASK_ROOT} --verbose

# Copy function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}
COPY mcp_tools.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler
CMD ["lambda_function.lambda_handler"]