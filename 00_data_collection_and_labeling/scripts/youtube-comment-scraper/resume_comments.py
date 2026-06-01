"""
Resume comment scraping without re-running YouTube API calls.
Reads video IDs from the existing metadata CSV and skips already-processed videos.
"""

import os
import pandas as pd
from tqdm import tqdm
import time
import json
import re
import requests

# =====================
# CONFIG (same as original)
# =====================

COMMENTS_FILE = "youtube_war_comments.csv"
METADATA_FILE = "youtube_video_metadata.csv"
PROCESSED_FILE = "processed_videos.txt"

# =====================
# WEB SCRAPER (copied from original - no API needed)
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
        try:
            response = session.post(url, params={'key': ytcfg['INNERTUBE_API_KEY']}, json=data, timeout=30)
            if response.status_code == 200:
                return response.json()
            if response.status_code in [403, 413]:
                return {}
            else:
                time.sleep(sleep)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, 
                requests.exceptions.ReadTimeout) as e:
            print(f"    Network error, retrying in {sleep}s... ({e.__class__.__name__})")
            time.sleep(sleep)
    return {}


def download_comments(video_id, sort_by=1):
    """
    Web scrape ALL comments from a video (NO API QUOTA!)
    sort_by: 0 = popular, 1 = recent
    """
    YOUTUBE_VIDEO_URL = f'https://www.youtube.com/watch?v={video_id}'
    session = requests.Session()
    session.headers['User-Agent'] = USER_AGENT
    
    try:
        response = session.get(YOUTUBE_VIDEO_URL, timeout=30)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        print(f"    Connection error: {e.__class__.__name__}")
        return []

    if 'uxe=' in response.request.url:
        session.cookies.set('CONSENT', 'YES+cb', domain='.youtube.com')
        response = session.get(YOUTUBE_VIDEO_URL, timeout=30)

    html = response.text
    ytcfg = json.loads(regex_search(html, YT_CFG_RE, default='{}'))
    if not ytcfg:
        return []

    data = json.loads(regex_search(html, YT_INITIAL_DATA_RE, default='{}'))

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
        return set(line.strip() for line in f if line.strip())


def save_processed_video(video_id):
    with open(PROCESSED_FILE, "a", encoding="utf-8") as f:
        f.write(video_id + "\n")


def append_comments(rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not os.path.exists(COMMENTS_FILE)
    df.to_csv(COMMENTS_FILE, mode="a", header=header, index=False, encoding="utf-8")


# =====================
# MAIN - RESUME COMMENT SCRAPING ONLY
# =====================

def main():
    # Load video IDs from existing metadata CSV (no API call needed)
    if not os.path.exists(METADATA_FILE):
        print(f"❌ Metadata file '{METADATA_FILE}' not found. Run the original script first.")
        return

    meta_df = pd.read_csv(METADATA_FILE)
    all_video_ids = meta_df['video_id'].unique().tolist()
    print(f"Total unique videos in metadata: {len(all_video_ids)}")

    # Load already-processed videos
    processed = load_processed_videos()
    print(f"Already processed: {len(processed)}")

    # Filter to unprocessed videos only
    to_process = [v for v in all_video_ids if v not in processed]
    print(f"Remaining to scrape: {len(to_process)}")

    if not to_process:
        print("✅ All videos already processed!")
        return

    print("\nScraping comments (no API quota used!)...")
    total_comments = 0
    errors = 0

    for vid in tqdm(to_process):
        try:
            comments = download_comments(vid)
            if comments:
                append_comments(comments)
                total_comments += len(comments)
                print(f"  ✓ {vid}: {len(comments)} comments")
            else:
                print(f"  ⚠ {vid}: No comments (disabled or error)")
            save_processed_video(vid)
            time.sleep(0.5)
        except KeyboardInterrupt:
            print(f"\n\n⏸  Interrupted! Progress saved.")
            print(f"   Processed so far: {len(load_processed_videos())} videos")
            print(f"   Comments collected: {total_comments}")
            print(f"   Run this script again to resume.")
            return
        except Exception as e:
            errors += 1
            print(f"  ✗ {vid}: Error - {e}")
            save_processed_video(vid)  # Skip on retry
            continue

    print(f"\n✅ Comment scraping completed!")
    print(f"   Total videos processed: {len(load_processed_videos())}")
    print(f"   Comments collected this run: {total_comments}")
    if errors:
        print(f"   Errors: {errors}")


if __name__ == "__main__":
    main()
