#!/bin/bash

# Check if AWS CLI is configured
if ! aws sts get-caller-identity &> /dev/null; then
    echo "AWS CLI is not configured. Please run 'aws configure' first."
    exit 1
fi

# Create the Lambda function
echo "Creating Lambda function..."
aws lambda create-function \
    --function-name AutomatedTikToks-TranscriptFetcher \
    --runtime python3.9 \
    --handler transcript_fetch.lambda_handler \
    --role arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/lambda_basic_execution \
    --zip-file fileb://transcript_fetch_lambda.zip \
    --timeout 30 \
    --memory-size 256

echo "Deployment complete!" 