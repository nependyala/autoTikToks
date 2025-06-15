#!/bin/bash

# Create IAM user
echo "Creating IAM user..."
aws iam create-user --user-name lambda-deployer

# Create access key for the user
echo "Creating access key..."
aws iam create-access-key --user-name lambda-deployer

# Attach necessary policies
echo "Attaching policies..."
aws iam attach-user-policy --user-name lambda-deployer --policy-arn arn:aws:iam::aws:policy/AWSLambda_FullAccess
aws iam attach-user-policy --user-name lambda-deployer --policy-arn arn:aws:iam::aws:policy/IAMFullAccess

echo "IAM user setup complete!"
echo "Please save the AccessKeyId and SecretAccessKey that were output above." 