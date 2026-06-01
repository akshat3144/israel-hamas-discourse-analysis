import os
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tqdm import tqdm
import time
import json
import re
import requests

# =====================
# CONFIG
# =====================

API_KEY = "AIzaSyBhaCRR6AftskQ69ZX37Hu9bZ_JPYkpjrk"

COMMENTS_FILE = "youtube_war_comments.csv"
METADATA_FILE = "youtube_video_metadata.csv"
PROCESSED_FILE = "processed_videos.txt"

PUBLISHED_AFTER = "2023-10-07T00:00:00Z"
PUBLISHED_BEFORE = "2024-05-07T23:59:59Z"

QUERIES = [
    # High-engagement core topics
    "israel hamas war",
    "gaza war",
    "israel gaza latest",
    
    # Critical events (high specificity, high engagement)
    "october 7 israel attack",
    "gaza hospital attack",
    "rafah offensive",
    
    # Humanitarian (emotional, high comment engagement)
    "gaza humanitarian crisis",
    "civilian casualties gaza",
    "gaza bombing",
    
    # Political/diplomatic (current, debated)
    "israel ceasefire talks",
    "gaza ceasefire",
    
    # Military/operational (specific, followers)
    "idf in gaza",
    "rocket attacks israel",
    
    # Aid/support (polarizing, high engagement)
    "aid to gaza",
]

youtube = build("youtube", "v3", developerKey=API_KEY)

# =====================
# WEB SCRAPER SETUP (NO API QUOTA!)
# =====================

USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36'
YT_CFG_RE = r'ytcfg\.set\s*\(\s*({.+?})\s*\)\s*;'
YT_INITIAL_DATA_RE = r'(?:window\s*\[\s*["\']ytInitialData["\']\s*\]|ytInitialData)\s*=\s*({.+?})\s*;\s*(?:var\s+meta|</script|\n)'


def regex_search(text, pattern, group=1, default=None):
    match = re.search(pattern, text)
    return match.group(group) if match else default


def search_dict(partial, search_key):
    stack = [partial]
    while stack:
        current_item = stack.pop()
        if isinstance(current_item, dict):
            for key, value in current_item.items():
                if key == search_key:
                    yield value
                else:
                    stack.append(value)
        elif isinstance(current_item, list):
            for value in current_item:
                stack.append(value)


def ajax_request(session, endpoint, ytcfg, retries=5, sleep=20):
    url = 'https://www.youtube.com' + endpoint['commandMetadata']['webCommandMetadata']['apiUrl']
    data = {'context': ytcfg['INNERTUBE_CONTEXT'],
            'continuation': endpoint['continuationCommand']['token']}

    for _ in range(retries):
        response = session.post(url, params={'key': ytcfg['INNERTUBE_API_KEY']}, json=data)
        if response.status_code == 200:
            return response.json()
        if response.status_code in [403, 413]:
            return {}
        else:
            time.sleep(sleep)


def download_comments(video_id, sort_by=1):
    """
    Web scrape ALL comments from a video (NO API QUOTA!)
    sort_by: 0 = popular, 1 = recent
    """
    YOUTUBE_VIDEO_URL = f'https://www.youtube.com/watch?v={video_id}'
    session = requests.Session()
    session.headers['User-Agent'] = USER_AGENT
    response = session.get(YOUTUBE_VIDEO_URL)

    if 'uxe=' in response.request.url:
        session.cookies.set('CONSENT', 'YES+cb', domain='.youtube.com')
        response = session.get(YOUTUBE_VIDEO_URL)

    html = response.text
    ytcfg = json.loads(regex_search(html, YT_CFG_RE, default=''))
    if not ytcfg:
        return []

    data = json.loads(regex_search(html, YT_INITIAL_DATA_RE, default=''))

    section = next(search_dict(data, 'itemSectionRenderer'), None)
    renderer = next(search_dict(section, 'continuationItemRenderer'), None) if section else None
    if not renderer:
        return []  # Comments disabled

    needs_sorting = sort_by != 0
    continuations = [renderer['continuationEndpoint']]
    
    comments = []
    
    while continuations:
        continuation = continuations.pop()
        response = ajax_request(session, continuation, ytcfg)

        if not response:
            break
        if list(search_dict(response, 'externalErrorMessage')):
            break

        if needs_sorting:
            sort_menu = next(search_dict(response, 'sortFilterSubMenuRenderer'), {}).get('subMenuItems', [])
            if sort_by < len(sort_menu):
                continuations = [sort_menu[sort_by]['serviceEndpoint']]
                needs_sorting = False
                continue

        actions = list(search_dict(response, 'reloadContinuationItemsCommand')) + \
                  list(search_dict(response, 'appendContinuationItemsAction'))
        
        for action in actions:
            for item in action.get('continuationItems', []):
                if action['targetId'] in ('comments-section', 'engagement-panel-comments-section'):
                    continuations[:0] = [ep for ep in search_dict(item, 'continuationEndpoint')]
                if action['targetId'].startswith('comment-replies-item') and 'continuationItemRenderer' in item:
                    continuations.append(next(search_dict(item, 'buttonRenderer'))['command'])

        # --- Legacy extraction (commentRenderer) ---
        for comment in reversed(list(search_dict(response, 'commentRenderer'))):
            comments.append({
                'video_id': video_id,
                'comment_id': comment['commentId'],
                'parent_comment_id': None,
                'comment_text': ''.join([c['text'] for c in comment['contentText'].get('runs', [])]),
                'comment_published_at': comment['publishedTimeText']['runs'][0]['text'],
                'comment_updated_at': comment['publishedTimeText']['runs'][0]['text'],
                'author': comment.get('authorText', {}).get('simpleText', ''),
                'votes': comment.get('voteCount', {}).get('simpleText', '0'),
            })

        # --- New extraction (commentEntityPayload via mutations) ---
        for mutation_batch in search_dict(response, 'mutations'):
            if not isinstance(mutation_batch, list):
                continue
            for m in mutation_batch:
                if not isinstance(m, dict):
                    continue
                ce = m.get('payload', {}).get('commentEntityPayload')
                if not ce:
                    continue
                props = ce.get('properties', {})
                author_info = ce.get('author', {})
                toolbar_info = ce.get('toolbar', {})
                content = props.get('content', {})
                text = content.get('content', '') if isinstance(content, dict) else str(content)
                comments.append({
                    'video_id': video_id,
                    'comment_id': props.get('commentId', ''),
                    'parent_comment_id': None,
                    'comment_text': text,
                    'comment_published_at': props.get('publishedTime', ''),
                    'comment_updated_at': props.get('publishedTime', ''),
                    'author': author_info.get('displayName', ''),
                    'votes': toolbar_info.get('likeCountNotliked', '0'),
                })

        time.sleep(0.1)

    return comments


# =====================
# PERSISTENT STORAGE
# =====================

def load_processed_videos():
    if not os.path.exists(PROCESSED_FILE):
        return set()
    with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f)


def save_processed_video(video_id):
    with open(PROCESSED_FILE, "a", encoding="utf-8") as f:
        f.write(video_id + "\n")


# =====================
# SEARCH VIDEOS (USES API)
# =====================

def search_videos(query, max_pages=6):
    video_ids = []
    token = None

    for _ in range(max_pages):
        try:
            res = youtube.search().list(
                part="id",
                q=query,
                type="video",
                order="viewCount",  # Most viewed = most comments
                publishedAfter=PUBLISHED_AFTER,
                publishedBefore=PUBLISHED_BEFORE,
                maxResults=50,
                pageToken=token
            ).execute()

            for item in res["items"]:
                video_ids.append(item["id"]["videoId"])

            token = res.get("nextPageToken")
            if not token:
                break
                
        except HttpError as e:
            if e.resp.status == 403 and 'quota' in str(e).lower():
                print(f"\n⚠️  Quota exceeded during search for '{query}'")
                raise
            else:
                print(f"Error searching '{query}': {e}")
                break

    return video_ids


# =====================
# VIDEO METADATA (USES API)
# =====================

def get_video_metadata(batch_ids):
    """
    Fetch metadata for up to 50 videos at once.
    """
    try:
        res = youtube.videos().list(
            part="snippet",
            id=",".join(batch_ids)
        ).execute()

        rows = []

        for item in res["items"]:
            s = item["snippet"]

            rows.append({
                "video_id": item["id"],
                "title": s.get("title"),
                "description": s.get("description"),
                "channel_id": s.get("channelId"),
                "channel_title": s.get("channelTitle"),
                "published_at": s.get("publishedAt")
            })

        return rows
        
    except HttpError as e:
        if e.resp.status == 403 and 'quota' in str(e).lower():
            print(f"\n⚠️  Quota exceeded during metadata fetch")
            raise
        else:
            print(f"Error fetching metadata: {e}")
            return []


def append_metadata(rows):
    if not rows:
        return

    df = pd.DataFrame(rows)
    header = not os.path.exists(METADATA_FILE)

    df.to_csv(
        METADATA_FILE,
        mode="a",
        header=header,
        index=False,
        encoding="utf-8"
    )


# =====================
# COMMENTS (WEB SCRAPING - NO API!)
# =====================

def append_comments(rows):
    if not rows:
        return

    df = pd.DataFrame(rows)
    header = not os.path.exists(COMMENTS_FILE)

    df.to_csv(
        COMMENTS_FILE,
        mode="a",
        header=header,
        index=False,
        encoding="utf-8"
    )


# =====================
# MAIN
# =====================

def main():

    processed = load_processed_videos()
    print(f"Already processed: {len(processed)}")

    # ---- SEARCH (USES API) ----
    all_video_ids = set()

    try:
        for q in QUERIES:
            print(f"Searching: {q}")
            vids = search_videos(q)
            all_video_ids.update(vids)
            time.sleep(1)

        to_process = [v for v in all_video_ids if v not in processed]
        print(f"New videos: {len(to_process)}")

        # ---- METADATA (USES API) ----
        print("Fetching metadata...")
        for i in range(0, len(to_process), 50):
            batch = to_process[i:i+50]
            meta_rows = get_video_metadata(batch)
            append_metadata(meta_rows)
            time.sleep(0.2)

        # ---- COMMENTS (WEB SCRAPING - NO API QUOTA!) ----
        print("\nScraping comments (no API quota used!)...")
        for vid in tqdm(to_process):
            try:
                comments = download_comments(vid)
                if comments:
                    append_comments(comments)
                    print(f"  ✓ {vid}: {len(comments)} comments")
                else:
                    print(f"  ⚠ {vid}: No comments (disabled or error)")
                save_processed_video(vid)
                time.sleep(0.5)  # Be nice to YouTube
            except Exception as e:
                print(f"  ✗ {vid}: Error - {e}")
                save_processed_video(vid)  # Save anyway to skip on retry
                continue
            
    except HttpError as e:
        if e.resp.status == 403 and 'quota' in str(e).lower():
            print("\n" + "="*60)
            print("❌ YouTube API quota exceeded!")
            print("="*60)
            print(f"✓ Progress saved: {len(load_processed_videos())} videos processed")
            print(f"✓ Found {len(all_video_ids)} unique videos total")
            print("\nTo continue:")
            print("  1. Wait 24 hours for quota reset (resets at midnight PT)")
            print("  2. Run this script again - it will resume from where it left off")
            print("\nNote: Comment scraping uses NO quota, only video search does!")
            print("="*60)
            return
        else:
            raise
    
    print(f"\n✅ Scraping completed successfully!")
    print(f"Total videos processed: {len(load_processed_videos())}")


if __name__ == "__main__":
    main()
