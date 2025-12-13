# Israel–Hamas War Discourse Analysis: Data Collection System

This project implements a comprehensive data collection system for analyzing online discourse about the **Israel–Hamas war** across **Reddit**, **YouTube**, and **Telegram**. The methodology is based on recent academic research examining social media narratives, sentiment patterns, and information dissemination during the conflict.

---

## Research Basis

This implementation is inspired by and extends methodologies from:

- **"Israel–Hamas war through Telegram, Reddit and Twitter"**\_Despoina Antonakaki & Sotiris Ioannidis (2025)\_Cross-platform analysis of discourse patterns and information flow
- **"Sentiment analysis of the Hamas–Israel war on YouTube"** _(2025)_
  Sentiment dynamics and public opinion formation through video comments

This project enables researchers to collect, analyze, and compare public discourse across multiple social media platforms to understand narrative formation, sentiment evolution, and topic prevalence during the ongoing conflict.

---

## Key Features

- **Multi-Platform Coverage** — Collects data from Reddit, YouTube, and Telegram simultaneously
- **Targeted Collection** — Keyword-based filtering for relevant conflict discussions
- **Date Range Filtering** — Focus on specific time periods (Oct 2023 onwards)
- **Rich Metadata** — Captures engagement metrics, timestamps, and user information
- **Automated Processing** — Batch collection with built-in rate limiting
- **Structured Output** — Clean XLSX format ready for analysis
- **Privacy-Focused** — Public data only, respects platform ToS
- **Analysis-Ready** — Compatible with sentiment analysis, topic modeling, and NLP tools

---

## Platforms Covered

- **Reddit (Public JSON API)** — Latest and historical posts from conflict-related subreddits
- **YouTube (YouTube Data API v3)** — Video metadata + comments from verified news channels, filtered by keywords
- **Telegram (Telethon)** — Public channel messages within a specific date range

---

## Setup Instructions

### 1. Install Required Libraries

```bash
pip install -r requirements.txt
```

**Key dependencies:**

- `pandas`, `numpy` — Data processing
- `telethon` — Telegram data collection
- `praw` — Reddit API wrapper
- `google-api-python-client` — YouTube Data API
- `python-dotenv` — Environment variable management
- Additional libraries for sentiment analysis and topic modeling (optional)

---

### 2. API Credentials & Environment Setup

Create a `.env` file in your project root:

```env
# YouTube Data API v3
YOUTUBE_API_KEY=your_youtube_api_key

# Telegram API Credentials
TELEGRAM_API_ID=your_api_id
TELEGRAM_API_HASH=your_api_hash
TELEGRAM_PHONE=+1234567890
```

**How to get API credentials:**

- **YouTube:** Get your API key from [Google Cloud Console](https://console.cloud.google.com/) → Enable YouTube Data API v3
- **Telegram:** Register your app at [my.telegram.org](https://my.telegram.org) → API Development Tools
- **Reddit:** No credentials needed — uses Reddit's public JSON endpoints

---

## File Structure

```
data_collector/
├── reddit_collector.py          # Reddit data collection (Public JSON)
├── youtube_collector.py         # YouTube data collection (API + keywords)
├── telegram_collector.py        # Telegram data collection (Telethon + date filter)
├── requirements.txt             # Python dependencies
├── .env                         # API credentials (not tracked in git)
├── README.md                    # Documentation
├── docs/                        # Additional documentation
└── collected_data/              # Output directory (auto-created)
    ├── reddit.xlsx
    ├── youtube.xlsx
    └── telegram.xlsx
```

---

## Usage Guide

### Step 1 — Reddit Data Collection

```bash
python reddit_collector.py
```

**What it does:**

- Searches targeted subreddits for conflict-related keywords
- Fetches posts and associated comments using Reddit's public JSON API
- Collects comprehensive metadata including scores, timestamps, and author information
- Outputs data to `collected_data/reddit.xlsx`

**Features:**

- No API key required
- Built-in rate limiting (1-second delay)
- Captures both posts and detailed comment threads

---

### Step 2 — YouTube Data Collection

```bash
python youtube_collector.py
```

**What it does:**

- Searches videos from verified news channels using keywords
- Filters by date range and conflict-related terms
- Extracts video metadata and comment threads
- Outputs data to `collected_data/youtube.xlsx`

**Features:**

- Requires YouTube Data API v3 key
- Keyword-based filtering
- Captures video details and comment engagement metrics

---

### Step 3 — Telegram Data Collection

```bash
python telegram_collector.py
```

**What it does:**

- Connects to Telegram using Telethon library
- Collects messages from predefined public channels
- Filters by date range (Oct 1, 2023 → Mar 31, 2024)
- Limits collection to 3,000 messages per channel
- Outputs data to `collected_data/telegram.xlsx`

**Features:**

- Requires Telegram API credentials
- Interactive phone authentication (first run only)
- Creates persistent session files for subsequent runs
- Captures message metadata including views, forwards, and replies

---

## Data Collection Details

### Reddit

**Subreddits:**
`Palestine`, `Israel`, `IsraelPalestine`, `worldnews`, `news`, `MiddleEastNews`, `geopolitics`

**Keywords:**
`Palestine`, `Gaza`, `Israel`, `Hamas`, `IDF`, `West Bank`, `Israeli occupation`, `Gaza Strip`

**Fields:**

| Field               | Description              |
| ------------------- | ------------------------ |
| Unnamed: 0          | Row index                |
| comment_id          | Unique comment ID        |
| score               | Comment score (upvotes)  |
| self_text           | Comment text             |
| subreddit           | Source subreddit         |
| created_time        | UTC timestamp            |
| post_id             | Associated post ID       |
| author_name         | Comment author username  |
| controversiality    | Controversiality score   |
| user_is_verified    | Verified user status     |
| post_score          | Parent post score        |
| post_self_text      | Parent post body         |
| post_title          | Parent post title        |
| post_upvote_ratio   | Parent post upvote ratio |
| post_created_time   | Parent post timestamp    |
| clean_text_comments | Cleaned comment text     |
| clean_text_posts    | Cleaned post text        |

---

### YouTube

**Channels:** BBC News | Al Jazeera English | CNN | Reuters | WION

**Keywords:**
`Israel`, `Hamas`, `Palestine`, `Gaza`, `Conflict`, `War`, `Ceasefire`, `Jerusalem`, `Middle East`, `IDF`

**Fields:**

| Field          | Description        |
| -------------- | ------------------ |
| video_id       | YouTube video ID   |
| channel_name   | Source channel     |
| video_title    | Title              |
| published_date | Upload date        |
| description    | Video description  |
| comment_text   | Individual comment |
| comment_author | Comment author     |
| comment_date   | Comment timestamp  |
| like_count     | Comment likes      |
| reply_count    | Replies            |
| keyword        | Matched keyword    |

---

### Telegram

**Channels Monitored:**
`AlQassamBrigades`, `Aqsatvsat`, `Eyeonpalestine`, `FreePalestine2023`, `GazaNow`, `PalestineSolidarityBelgium`, `PalestineUpdates`, `PalestinianResistance`, `StopGazaGenocide`, `TIMESOFGAZA`, `TheJerusalemPost`, `bigolivr`, `gazaalanpa`, `gazaenglishupdates`, `haqqintel`, `palOnline`, `palestineonline`, `palestineresistance`, `resistancechain`

**Message Limit:** 3,000 messages per channel

**Date Range:** `2023-10-01 → 2024-03-31`

**Fields:**

| Field      | Description             |
| ---------- | ----------------------- |
| channel    | Source Telegram channel |
| message_id | Unique message ID       |
| date       | UTC timestamp           |
| text       | Message content         |
| views      | Number of views         |
| forwards   | Number of forwards      |
| replies    | Number of replies       |
| link       | Direct message URL      |

---

## Output Examples

All data is saved in XLSX (Excel) format for easy analysis. Below are example data structures:

### Reddit

```json
{
  "Unnamed: 0": 0,
  "comment_id": "abc123",
  "score": 42,
  "self_text": "This is a thoughtful comment about the conflict...",
  "subreddit": "worldnews",
  "created_time": "2024-11-15T10:30:00",
  "post_id": "xyz789",
  "author_name": "reddit_user",
  "controversiality": 0,
  "user_is_verified": false,
  "post_score": 512,
  "post_self_text": "Latest updates from Gaza...",
  "post_title": "Israel–Hamas conflict intensifies",
  "post_upvote_ratio": 0.85
}
```

### YouTube

```json
{
  "video_id": "xyz789",
  "channel_name": "BBC News",
  "video_title": "Israel–Hamas Conflict Update",
  "comment_text": "Praying for peace",
  "comment_date": "2024-11-15T10:30:00",
  "keyword": "Gaza"
}
```

### Telegram

```json
{
  "channel": "TimesOfGaza",
  "message_id": 12345,
  "date": "2024-10-12T09:15:00Z",
  "text": "Breaking: ceasefire discussions underway.",
  "views": 15800,
  "forwards": 120,
  "replies": 6,
  "link": "https://t.me/TimesOfGaza/12345"
}
```

---

## Next Steps — Analysis

1. **Sentiment Analysis** – `VADER`, `TextBlob`, or Hugging Face models
2. **Topic Modeling** – `LDA`, `BERTopic`, `Top2Vec`
3. **Entity Extraction** – Identify people, places, organizations
4. **Trend Analysis** – Measure narrative shifts over time
5. **Cross-Platform Comparison** – Contrast Reddit vs YouTube vs Telegram tone and reach

---

## Important Notes

### Rate Limits & Quotas

- **Reddit:** 1-second delay between requests (built-in rate limiting)
- **YouTube:** 10,000-unit daily quota limit (quota resets at midnight Pacific Time)
- **Telegram:** Respect Telegram's flood limits; avoid excessive requests

### Data Collection Guidelines

- **Public Data Only:** Collect only publicly available data
- **Terms of Service:** Respect each platform's ToS and API usage policies
- **Privacy:** Anonymize user information before analysis or publication
- **Research Ethics:** Use collected data for research and educational purposes only

### Session Files

- Telegram authentication creates session files (`.session`) for persistent login
- Keep these files secure and do not share them
- Add `*.session` to `.gitignore` to prevent accidental commits

---

## Troubleshooting

### Reddit Collection Issues

- **"No posts found"** — Try different keywords or subreddits
- **Rate limiting** — Script includes built-in delays; no action needed
- **Empty comments** — Some posts may have deleted comments

### YouTube Collection Issues

- **"Quota exceeded"** — YouTube API has 10,000 daily units; wait for reset
- **"Invalid API key"** — Verify your API key in `.env` and ensure YouTube Data API v3 is enabled
- **Limited results** — Adjust `max_results` parameter or expand keyword list

### Telegram Collection Issues

- **Authentication fails** — Ensure phone number format is correct (include country code)
- **"FloodWaitError"** — Telegram rate limit hit; wait before retrying
- **Session errors** — Delete `.session` files and re-authenticate
- **Channel not found** — Verify channel username (without @) is correct

---

## References

- _Israel–Hamas war through Telegram, Reddit and Twitter_ — Despoina Antonakaki & Sotiris Ioannidis (2025)
- _Sentiment analysis of the Hamas–Israel war on YouTube_ — arXiv (2025)
