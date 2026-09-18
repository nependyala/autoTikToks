# AWS Lambda: YouTube transcript fetch (yt-dlp)

This folder packages a Lambda that downloads English auto-captions for a YouTube URL.

## Status

**Experimental.** `response.json` records a failed test invoke (yt-dlp YouTube bot check / Lambda read-only filesystem cache issues). Prefer the local `video_creation/youtube_clips/transcript_fetch.py` path for day-to-day use.

## Source

- `transcript_fetch.py` — Lambda handler source (restored from the deployment package)
- `template.yaml` — AWS SAM function definition
- `shell_scripts/` — IAM, package, and deploy helpers
- Generated `*.zip` artifacts are gitignored; rebuild with the shell scripts

## Deploy (high level)

1. Configure AWS CLI credentials for your account.
2. Build a deployment zip with `shell_scripts/deploy_lambda.sh` or `package.sh`.
3. Deploy via SAM (`template.yaml`) or upload the zip in the console.
4. Handler: `transcript_fetch.lambda_handler` (see packaged module layout).
5. Suggested: ≥256MB memory, ≥30s timeout.

## Test event

```json
{
  "video_url": "https://www.youtube.com/watch?v=VIDEO_ID"
}
```

Expected success shape:

```json
{
  "statusCode": 200,
  "body": {
    "transcript": "..."
  }
}
```

YouTube may require cookies for some environments; cookie-updater scripts under `shell_scripts/` are related scaffolding and should be treated as incomplete unless you verify them end-to-end.
