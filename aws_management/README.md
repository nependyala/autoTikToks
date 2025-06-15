# AWS Lambda Transcript Fetcher

This directory contains the AWS Lambda function for fetching YouTube video transcripts.

## Files
- `transcript_fetch.py`: The Lambda function code
- `requirements.txt`: Python dependencies
- `deploy_lambda.sh`: Script to create the Lambda deployment package

## Deployment Instructions

1. Make sure you have Python 3.9+ and pip installed

2. Create the deployment package:
```bash
./deploy_lambda.sh
```

3. In the AWS Console:
   - Create a new Lambda function
   - Choose Python 3.9 or later as the runtime
   - Upload the generated `transcript_fetch_lambda.zip` file
   - Set the handler to `transcript_fetch.lambda_handler`
   - Configure the function with:
     - Memory: 256MB (minimum)
     - Timeout: 30 seconds
     - Basic Lambda execution role

## Testing the Lambda Function

Use this test event in the Lambda console:
```json
{
  "video_url": "https://www.youtube.com/watch?v=Au2glAZrWAo"
}
```

The function will return the transcript content in the response body:
```json
{
  "statusCode": 200,
  "body": {
    "transcript": "... transcript content ..."
  }
}
```

## Error Handling

The function returns appropriate error responses:
- 400: Missing video_url in the event
- 500: Any other error (e.g., failed to download transcript) 