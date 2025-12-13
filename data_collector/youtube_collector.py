"""
YouTube Data Collection Implementation
Collects video comments and metadata using YouTube Data API v3
Filtered by conflict-related keywords and date range
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict
from googleapiclient.discovery import build
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")


class YouTubeCollector:
    """Collects data (comments + metadata) from YouTube channels or videos"""

    def __init__(self, api_key: str, output_dir: str = "../data"):
        self.api_key = api_key
        self.youtube = build("youtube", "v3", developerKey=api_key)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        print("✓ Connected to YouTube Data API")

    def get_channel_videos(
        self, channel_id: str, max_results: int = 50, keywords: List[str] = None,
        start_date: str = None, end_date: str = None
    ) -> List[str]:
        """
        Retrieve video IDs from a specific channel filtered by keywords and date
        """
        video_ids = []
        keywords_query = " ".join(keywords) if keywords else ""
        published_after = f"{start_date}T00:00:00Z" if start_date else None
        published_before = f"{end_date}T23:59:59Z" if end_date else None

        request = self.youtube.search().list(
            q=keywords_query,
            channelId=channel_id,
            part="id,snippet",
            maxResults=max_results,
            order="date",
            type="video",
            publishedAfter=published_after,
            publishedBefore=published_before
        )

        response = request.execute()
        for item in response.get("items", []):
            video_ids.append(item["id"]["videoId"])

        print(f"✓ Found {len(video_ids)} relevant videos for channel ID: {channel_id}")
        return video_ids

    def get_video_comments(self, video_id: str, max_comments: int = 500) -> List[Dict]:
        """Collect comments for a given video"""
        comments = []
        try:
            request = self.youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                textFormat="plainText"
            )
            while request and len(comments) < max_comments:
                response = request.execute()
                for item in response.get("items", []):
                    snippet = item["snippet"]["topLevelComment"]["snippet"]
                    comment_data = {
                        "id": item["id"],
                        "video id": video_id,
                        "author": snippet.get("authorDisplayName"),
                        "author id": snippet.get("authorChannelId", {}).get("value", ""),
                        "text": snippet.get("textDisplay"),
                        "label": "",
                        "annotator notes": ""
                    }
                    comments.append(comment_data)

                if "nextPageToken" in response:
                    request = self.youtube.commentThreads().list(
                        part="snippet",
                        videoId=video_id,
                        maxResults=100,
                        pageToken=response["nextPageToken"],
                        textFormat="plainText"
                    )
                else:
                    break

            print(f"  ✓ Collected {len(comments)} comments for video {video_id}")
        except Exception as e:
            print(f"✗ Error fetching comments for {video_id}: {e}")

        return comments

    def collect_from_channels(
        self, channel_ids: Dict[str, str], max_videos_per_channel: int = 10,
        max_comments_per_video: int = 500, keywords: List[str] = None,
        start_date: str = None, end_date: str = None
    ) -> List[Dict]:
        """Collect comments from multiple channels filtered by keywords and date"""
        all_comments = []
        for channel_name, channel_id in channel_ids.items():
            print(f"\n📺 Collecting from {channel_name}...")
            video_ids = self.get_channel_videos(
                channel_id,
                max_results=max_videos_per_channel,
                keywords=keywords,
                start_date=start_date,
                end_date=end_date
            )

            for vid in video_ids:
                video_comments = self.get_video_comments(vid, max_comments_per_video)
                all_comments.extend(video_comments)

        print(f"\n✓ Total comments collected: {len(all_comments)}")
        return all_comments

    def save_data(self, data: List[Dict], filename: str = None):
        """Save collected data as XLSX"""
        if not filename:
            filename = f"youtube_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        xlsx_path = os.path.join(self.output_dir, f"{filename}.xlsx")

        pd.DataFrame(data).to_excel(xlsx_path, index=False, engine='openpyxl')
        print(f"\n✓ Data saved:")
        print(f"  XLSX → {xlsx_path}")
        print(f"  Total comments: {len(data)}")


def main():
    """Collect YouTube comments on Israel–Hamas War from news channels"""
    print("=" * 60)
    print("YOUTUBE DATA COLLECTION (Keyword + Date Filter)")
    print("=" * 60)

    collector = YouTubeCollector(API_KEY)

    CHANNEL_IDS = {
        "BBC": "UC16niRr50-MSBwiO3YDb3RA",
        "Aljazeera": "UCNye-wNBqNL5ZzHSJj3l8Bg",
        "CNN": "UCupvZG-5ko_eiXAupbDfxWw",
        "WION": "UC_gUM8rL-Lrg6O3adPW9K1g",
        "Reuters": "UCZLZ8Jjx_RN2CXloOmgTHVg"
    }

    KEYWORDS = [
        "Israel",
        "Palestine",
        "Gaza",
        "Hamas",
        "IDF",
        "Middle East",
        "Conflict",
        "War",
        "Ceasefire",
        "Jerusalem"
    ]
    START_DATE = "2023-10-01"
    END_DATE = "2024-03-31"

    comments = collector.collect_from_channels(
        channel_ids=CHANNEL_IDS,
        max_videos_per_channel=10,
        max_comments_per_video=500,
        keywords=KEYWORDS,
        start_date=START_DATE,
        end_date=END_DATE
    )

    collector.save_data(comments, filename="youtube")

    print("\n" + "=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    # Before running:
    # pip install google-api-python-client python-dotenv pandas
    # and create a .env file with: YOUTUBE_API_KEY=your_api_key
    main()
