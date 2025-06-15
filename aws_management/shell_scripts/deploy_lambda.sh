#!/bin/bash

# Create a temporary directory for building the package
mkdir -p lambda_package
cd lambda_package

# Install dependencies
pip install -r ../requirements.txt -t .

# Copy the Lambda function code from video_creation directory
cp ../video_creation/transcript_fetch.py .

# Create the deployment package
zip -r ../transcript_fetch_lambda.zip .

# Clean up
cd ..
rm -rf lambda_package

echo "Deployment package created: transcript_fetch_lambda.zip"
echo "You can now upload this package to AWS Lambda" 