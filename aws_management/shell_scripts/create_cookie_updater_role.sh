#!/bin/bash

# Create IAM role for the cookie updater Lambda
aws iam create-role \
    --role-name AutomatedTikToks-CookieUpdater \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {
                "Service": "lambda.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }]
    }'

# Attach basic Lambda execution policy
aws iam attach-role-policy \
    --role-name AutomatedTikToks-CookieUpdater \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

# Create and attach S3 access policy
aws iam put-role-policy \
    --role-name AutomatedTikToks-CookieUpdater \
    --policy-name S3Access \
    --policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::automated-tiktoks",
                "arn:aws:s3:::automated-tiktoks/*"
            ]
        }]
    }'

echo "Role created and policies attached successfully!" 