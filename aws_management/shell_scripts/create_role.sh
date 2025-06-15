#!/bin/bash

# Create the IAM role
echo "Creating IAM role..."
aws iam create-role \
    --role-name lambda_basic_execution \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "lambda.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }'

# Attach the basic Lambda execution policy
echo "Attaching execution policy..."
aws iam attach-role-policy \
    --role-name lambda_basic_execution \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

echo "Role creation complete!" 