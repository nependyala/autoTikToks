# autoTikToks

Personal tooling for creating short-form vertical video and preparing CapCut/TikTok posting workflows.

This repo is a **collection of scripts and pipelines**, not a single packaged app or hosted SaaS. Two content paths are supported:

1. **YouTube → short clip** — fetch captions, use Gemini to pick viral-style moments, download a segment, then convert to 9:16 with face-aware cropping, scene/cut detection, and optional TikTok-style captions.
2. **AI-generated ASMR clips** — OpenAI builds a Veo 3 prompt (glass fruit slicing), fal.ai generates an 8s 9:16 video, then schedule metadata can be written for CapCut posting.

> Sole author in git history: Neil Pendyala. Status: working personal automation toolkit / prototype. CapCut upload automation and AWS Lambda pieces are partial.

## Repo layout

```
autoTikToks/
├── video_creation/
│   ├── youtube_clips/     # Transcript → moment pick → clip download → edit → caption
│   └── ai_clips/          # OpenAI prompt gen + fal.ai Veo 3 generation
├── video_posting/         # CapCut Selenium login, schedule CSV helpers, sample clip slots
├── aws_management/        # SAM template + scripts for a YouTube transcript Lambda
├── requirements.txt
└── .env.example
```

## End-to-end workflows

### A) YouTube talking-head → TikTok-ready clip

```
YouTube URL
   → transcript_fetch.py   (yt-dlp English auto-subs)
   → decide_clip.py        (Gemini 2.0 Flash → timestamped "viral moments")
   → clip_fetch.py         (yt-dlp + ffmpeg trim)
   → edit.py               (9:16, hard cuts, face tracking / dynamic crop, finalize)
   → caption.py            (optional WhisperX word-timed TikTok-style captions)
```

Orchestrator:

```bash
cd video_creation/youtube_clips
python master.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

`master.py` runs steps through editing (high quality, frame-perfect, dynamic face crop). Captioning is separate via `caption.py`.

Standalone edit:

```bash
python edit.py input.mp4 output.mp4 --frame-perfect --dynamic-face-crop --quality high --platform tiktok
```

### B) AI ASMR fruit clips → schedule

```
prompt_generation.py  (OpenAI picks fruit + color → Veo3 prompt)
   → video_generation.py  (fal.ai veo3 → temporary_files/video_clips/*.mp4)
   → create_schedule.py   (CSV: file, date/time, platform, caption)
   → capcut_login.py      (Selenium login / navigation experiments)
```

Schedule helpers accept platforms: `tiktok`, `instagram`, `youtube`. Sample schedule rows target **TikTok**.

### C) AWS transcript Lambda (experimental)

`aws_management/` packages a Lambda that runs yt-dlp for captions (`template.yaml`, deploy shell scripts). A checked-in `response.json` shows a failed invoke (YouTube bot check / read-only FS cache). Treat as unfinished cloud experiment, not a reliable production service.

## What each area does

| Area | Does | Does not |
| --- | --- | --- |
| `youtube_clips/` | Caption fetch, LLM moment selection, clip download, CV-based 9:16 editing, optional captions | Host a web UI; auto-post |
| `ai_clips/` | Prompt + Veo3 generation for short ASMR clips | Edit/composite complex timelines |
| `video_posting/` | Schedule CSV tooling; CapCut login Selenium flow | Complete reliable multi-platform auto-poster (`capcut_scheduling.py` notes posting automation was removed) |
| `aws_management/` | SAM/IAM/package scripts for transcript Lambda | Proven always-on production deployment |

## Stack

- **Python** scripts (CLI)
- **yt-dlp** + **ffmpeg** — captions and clip download
- **Google Gemini** (`gemini-2.0-flash`) — viral moment detection from transcripts
- **OpenAI** — random fruit/color + Veo3 prompt text
- **fal.ai** (`fal-ai/veo3`) — text-to-video
- **OpenCV**, **face_recognition**, **PySceneDetect**, **scikit-image**, **MoviePy** — vertical edit pipeline
- **WhisperX** (optional, installed by `caption.py`) — word-level captions
- **Selenium** + Chrome — CapCut web login automation
- **AWS SAM / Lambda / IAM** scripts — transcript fetcher packaging

## Setup

```bash
git clone git@github.com:nependyala/autoTikToks.git
cd autoTikToks
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

mkdir -p secrets
cp .env.example secrets/.env
# Fill in API keys / CapCut credentials locally
```

System tools you may also need:

- `yt-dlp` and `ffmpeg` on `PATH`
- Chrome (for CapCut Selenium)
- Optional: WhisperX stack for `caption.py` (script can pip-install)

## Configuration

All scripts expect secrets in `secrets/.env` (gitignored). See `.env.example` for placeholders only:

| Variable | Used by |
| --- | --- |
| `GEMINI_API_KEY` | `youtube_clips/decide_clip.py` |
| `OPENAI_API_KEY` | `ai_clips/prompt_generation.py` |
| `FAL_AI_VEO3_API_KEY` | `ai_clips/video_generation.py` (mapped to `FAL_KEY`) |
| `CAPCUT_EMAIL` / `CAPCUT_PASSWORD` | `video_posting/capcut_login.py` |

## Example commands

```bash
# Full YouTube clip pipeline
python video_creation/youtube_clips/master.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Generate an AI clip prompt
python video_creation/ai_clips/prompt_generation.py

# Generate a Veo3 video (needs FAL key)
python video_creation/ai_clips/video_generation.py

# Build / extend a posting schedule CSV
python video_posting/helper_functions/create_schedule.py
```

## Current limitations

- Not one unified CLI or web product — modules are run separately
- CapCut **posting** automation is incomplete; scheduling helpers + login experiments exist
- AWS Lambda transcript fetch is brittle against YouTube bot checks (see `aws_management/response.json`)
- `caption.py` assumes macOS font paths for Arial Black
- `edit.py` class name (`MoistCritikalVideoProcessor`) reflects early talking-head niche targeting; pipeline is general vertical editing
- Large generated media/zips are gitignored; regenerate locally

## Security

Never commit `secrets/`, cookies, CapCut passwords, or API keys. Rotate any key that may have been exposed outside this repo.

## License

No license file present; treat as private/personal unless you add one.
