#!/bin/bash

# Container Build and Deploy Script for cdw-explore-advbot-template

# Variables

ACCOUNT_ID="471112976153"
REGION="us-east-1"
REPOSITORY_NAME="cdw-explore-advbot-template"
IMAGE_TAG="latest"
FUNCTION_NAME="cdw-explore-advbot-template"

# Config correct AWS Profile
export AWS_PROFILE=cdw-demo

# Disable Docker Trust
export DOCKER_CONTENT_TRUST=0

# Step 1: Create ECR repository
echo "Creating ECR repository..."
aws ecr create-repository \
    --repository-name $REPOSITORY_NAME \
    --region $REGION \
    --image-scanning-configuration scanOnPush=true \
    --encryption-configuration encryptionType=AES256 \
    2>/dev/null || echo "Repository already exists, continuing..."

# Step 2: Get ECR login token
echo "Logging into ECR..."
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

# Step 3: Build the Docker image
echo "Building Docker image..."
# Ensure buildx is available and create/use a builder that supports multi-platform builds
docker buildx create --name multiplatform --use --driver docker-container 2>/dev/null || docker buildx use multiplatform
docker buildx build --platform linux/amd64 --load -t $REPOSITORY_NAME:$IMAGE_TAG .

# Step 4: Tag the image for ECR
echo "Tagging image for ECR..."
docker tag $REPOSITORY_NAME:$IMAGE_TAG $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_NAME:$IMAGE_TAG

# Step 5: Push the image to ECR
echo "Pushing image to ECR..."
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_NAME:$IMAGE_TAG

# Step 6: Update existing Lambda function or create new one
echo "Checking if Lambda function exists..."
if aws lambda get-function --function-name $FUNCTION_NAME --region $REGION >/dev/null 2>&1; then
    echo "Function exists. Updating function code..."
    aws lambda update-function-code \
        --function-name $FUNCTION_NAME \
        --image-uri $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_NAME:$IMAGE_TAG \
        --region $REGION
    
    echo "Waiting for update to complete..."
    aws lambda wait function-updated --function-name $FUNCTION_NAME --region $REGION
    
    echo "Lambda function updated successfully!"
else
    echo "Function doesn't exist. Creating new Lambda function..."
    aws lambda create-function \
        --function-name $FUNCTION_NAME \
        --package-type Image \
        --code ImageUri=$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_NAME:$IMAGE_TAG \
        --role arn:aws:iam::$ACCOUNT_ID:role/cdw-explore-advbot-lambda-role \
        --timeout 90 \
        --memory-size 1024 \
        --architectures x86_64 \
        --description "CDW Explore Advisory Bot Template using LangChain and Bedrock (Container)" \
        --region $REGION
    
    echo "Lambda function created successfully!"
fi