#!/bin/bash

# Create a temporary directory
mkdir -p temp_package

# Create Python package structure
mkdir -p temp_package/transcript_fetch
touch temp_package/transcript_fetch/__init__.py

# Copy the Lambda function and cookies
cp ../video_creation/transcript_fetch.py temp_package/transcript_fetch/transcript_fetch.py
cp ../video_creation/temporary_files/cookies.txt temp_package/cookies.txt

# Install dependencies
pip install -r requirements.txt -t temp_package/

# Download yt-dlp binary
curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -o temp_package/yt-dlp
chmod +x temp_package/yt-dlp

# Create the deployment package
cd temp_package
zip -r ../transcript_fetch_lambda.zip .
cd ..

# Clean up
rm -rf temp_package

echo "Package created: transcript_fetch_lambda.zip" 