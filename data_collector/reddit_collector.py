"""
Reddit Data Collection via Reddit's Public JSON Endpoints
(No API credentials required, reliable alternative to Pushshift)
Filtered by conflict-related keywords and date range
"""

import requests
import time
from datetime import datetime
import json
import pandas as pd
import os
from typing import List, Dict


class RedditCollector:
    """
    Collects Reddit posts using Reddit's public JSON search endpoints
    """

    def __init__(self, output_dir: str = "../data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.base_url = "https://www.reddit.com"
        self.headers = {"User-Agent": "Mozilla/5.0 (Reddit Data Collector)"}
        print("✓ Connected to Reddit public JSON endpoints")

    def get_post_comments(self, permalink: str, post_data: Dict, subreddit: str) -> List[Dict]:
        """
        Fetch all comments for a specific post
        """
        comments = []
        try:
            url = f"{self.base_url}{permalink}.json"
            response = requests.get(url, headers=self.headers, timeout=30)
            
            if response.status_code != 200:
                return comments
            
            data = response.json()
            if len(data) < 2:
                return comments
            
            # Post data is in data[0], comments are in data[1]
            comment_data = data[1].get("data", {}).get("children", [])
            
            # Extract post metadata
            post_id = post_data.get("id")
            post_author = post_data.get("author")
            post_score = post_data.get("score", 0)
            post_title = post_data.get("title", "")
            post_selftext = post_data.get("selftext", "")
            post_upvote_ratio = post_data.get("upvote_ratio")
            post_created_utc = post_data.get("created_utc")
            post_created_time = datetime.utcfromtimestamp(post_created_utc).isoformat() if post_created_utc else ""
            
            # Process each comment
            for idx, comment in enumerate(comment_data):
                if comment.get("kind") != "t1":  # t1 is comment type
                    continue
                    
                c_data = comment.get("data", {})
                comment_author = c_data.get("author", "")
                
                # Get author details (when available)
                author_created_utc = c_data.get("author_created_utc")
                author_account_created = datetime.utcfromtimestamp(author_created_utc).isoformat() if author_created_utc else ""
                
                comment_record = {
                    "Unnamed: 0": idx,
                    "comment_id": c_data.get("id", ""),
                    "score": c_data.get("score", 0),
                    "self_text": c_data.get("body", ""),
                    "subreddit": subreddit,
                    "created_time": datetime.utcfromtimestamp(c_data.get("created_utc", 0)).isoformat() if c_data.get("created_utc") else "",
                    "post_id": post_id,
                    "author_name": comment_author,
                    "controversiality": c_data.get("controversiality", 0),
                    "user_is_verified": c_data.get("is_verified", False),
                    "user_account_created_time": author_account_created,
                    "user_total_karma": "",  # Not available in public API
                    "post_score": post_score,
                    "post_self_text": post_selftext,
                    "post_title": post_title,
                    "post_upvote_ratio": post_upvote_ratio,
                    "post_created_time": post_created_time,
                    "clean_text_comments": c_data.get("body", ""),
                    "clean_text_posts": post_selftext,
                    "Label": "",
                    "Annotator notes": ""
                }
                comments.append(comment_record)
            
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            print(f"✗ Error fetching comments from {permalink}: {e}")
        
        return comments

    def collect_comments(
        self,
        subreddits: List[str],
        keywords: List[str],
        max_posts_per_sub: int = 500,
        start_date: str = None,
        end_date: str = None
    ) -> List[Dict]:
        """
        Collect comments from multiple subreddits using Reddit's JSON endpoints (no API keys)
        Filters by conflict-related keywords and optional date range.
        """
        all_comments = []

        # Convert dates to timestamps
        start_timestamp = (
            int(datetime.strptime(start_date, "%Y-%m-%d").timestamp()) if start_date else None
        )
        end_timestamp = (
            int(datetime.strptime(end_date, "%Y-%m-%d").timestamp()) if end_date else None
        )

        for subreddit in subreddits:
            print(f"\nCollecting from r/{subreddit}...")
            collected = 0

            for keyword in keywords:
                after = None
                while collected < max_posts_per_sub:
                    url = f"{self.base_url}/r/{subreddit}/search.json"
                    params = {
                        "q": keyword,
                        "restrict_sr": "on",
                        "sort": "new",
                        "limit": "100",
                        "after": after
                    }

                    try:
                        response = requests.get(url, headers=self.headers, params=params, timeout=30)
                        if response.status_code != 200:
                            print(f"HTTP {response.status_code} for {subreddit} - {keyword}")
                            break

                        data = response.json().get("data", {})
                        posts = data.get("children", [])
                        if not posts:
                            break

                        for post in posts:
                            post_data = post["data"]
                            created_utc = post_data.get("created_utc")

                            # ✅ Filter by date range
                            if start_timestamp and created_utc < start_timestamp:
                                continue
                            if end_timestamp and created_utc > end_timestamp:
                                continue

                            # Fetch comments for this post
                            post_id = post_data.get("id")
                            permalink = post_data.get("permalink", "")
                            post_comments = self.get_post_comments(permalink, post_data, subreddit)
                            all_comments.extend(post_comments)

                        collected += len(posts)
                        print(f"  {keyword}: Collected {collected} posts so far...")
                        after = data.get("after")

                        if not after:
                            break

                        time.sleep(1)  # Avoid hitting rate limits

                    except Exception as e:
                        print(f"✗ Error collecting {subreddit}/{keyword}: {e}")
                        time.sleep(5)
                        break

            print(f"✓ Finished r/{subreddit} — total {collected} posts processed")

        # Deduplicate by comment_id
        unique_comments = {c["comment_id"]: c for c in all_comments if c.get("comment_id")}
        all_comments = list(unique_comments.values())

        print(f"\n✓ Total unique comments collected: {len(all_comments)}")
        return all_comments

    def save_data(self, data: List[Dict], filename: str = None):
        """
        Save collected data to XLSX
        """
        if not filename:
            filename = f"reddit_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        xlsx_path = os.path.join(self.output_dir, f"{filename}.xlsx")

        pd.DataFrame(data).to_excel(xlsx_path, index=False, engine='openpyxl')

        print(f"\n✓ Data saved:")
        print(f"  XLSX → {xlsx_path}")
        print(f"  Total records: {len(data)}")


def main():
    """
    Main execution
    """
    print("=" * 60)
    print("REDDIT DATA COLLECTION (Keyword + Date Filter)")
    print("=" * 60)

    collector = RedditCollector()

    SUBREDDITS = [
        "Palestine",
        "Israel",
        "IsraelPalestine",
        "worldnews",
        "news",
        "MiddleEastNews",
        "geopolitics"
    ]

    KEYWORDS = [
        "Palestine", "Gaza", "Israel", "Hamas",
        "IDF", "West Bank", "Gaza Strip", "Israeli occupation", "Middle East conflict"
    ]

    # Date range filter (Israel–Hamas war period)
    START_DATE = "2023-10-01"
    END_DATE = "2024-03-31"

    comments = collector.collect_comments(
        SUBREDDITS,
        KEYWORDS,
        max_posts_per_sub=2000,
        start_date=START_DATE,
        end_date=END_DATE
    )

    collector.save_data(comments, filename="reddit")

    print("\n" + "=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
