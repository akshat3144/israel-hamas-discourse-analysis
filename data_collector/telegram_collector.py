"""
Telegram Data Collection Script (with Date Filter and Message Limit)
Reads credentials from .env file
"""

import os
import json
import pandas as pd
from datetime import datetime, timezone
from telethon.sync import TelegramClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_ID = int(os.getenv("TELEGRAM_API_ID"))
API_HASH = os.getenv("TELEGRAM_API_HASH")
PHONE = os.getenv("TELEGRAM_PHONE")

# Channels to collect from
CHANNELS = [
    "AlQassamBrigades",
    "Aqsatvsat",
    "Eyeonpalestine",
    "FreePalestine2023",
    "GazaNow",
    "PalestineSolidarityBelgium",
    "PalestineUpdates",
    "PalestinianResistance",
    "StopGazaGenocide",
    "TIMESOFGAZA",
    "TheJerusalemPost",
    "bigolivr",
    "gazaalanpa",
    "gazaenglishupdates",
    "haqqintel",
    "palOnline",
    "palestineonline",
    "palestineresistance",
    "resistancechain"
]

OUTPUT_DIR = "collected_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

START_DATE = datetime(2023, 10, 1, tzinfo=timezone.utc)
END_DATE = datetime(2024, 3, 31, tzinfo=timezone.utc)
MESSAGE_LIMIT = 3000  # per channel

def collect_channel_messages(client, channel, start_date, end_date, limit):
    """
    Collect messages from a Telegram channel within date range
    """
    messages = []
    try:
        entity = client.get_entity(channel)
        count = 0

        for i, msg in enumerate(client.iter_messages(entity, limit=limit)):
            if msg.date is None:
                continue
            if msg.date.replace(tzinfo=timezone.utc) < start_date:
                break
            if msg.date.replace(tzinfo=timezone.utc) > end_date:
                continue

            messages.append({
                "channel_name": channel,
                "message_id": msg.id,
                "date": msg.date.isoformat(),
                "text": msg.text or "",
                "views": msg.views,
                "forwards": msg.forwards,
                "replies": msg.replies.replies if msg.replies else 0,
                "media": bool(msg.media)
            })

            count += 1
            if count % 500 == 0:
                print(f"  ...processed {count} messages from {channel}")

        print(f"  ✓ Collected {len(messages)} messages from {channel}")

    except Exception as e:
        print(f"✗ Error collecting from {channel}: {e}")

    return messages


def main():
    print("=" * 60)
    print("TELEGRAM DATA COLLECTION (Date Filter + Limit Enabled)")
    print("=" * 60)

    with TelegramClient("telegram_session", API_ID, API_HASH) as client:
        client.start(phone=PHONE)
        print("✓ Connected to Telegram API\n")
        print(f"Collecting messages between {START_DATE.date()} and {END_DATE.date()} from {len(CHANNELS)} channels...\n")

        all_messages = []
        for channel in CHANNELS:
            print(f"📢 Collecting from: {channel}")
            channel_msgs = collect_channel_messages(client, channel, START_DATE, END_DATE, MESSAGE_LIMIT)
            all_messages.extend(channel_msgs)

        # Save results
        xlsx_path = os.path.join(OUTPUT_DIR, "telegram.xlsx")

        pd.DataFrame(all_messages).to_excel(xlsx_path, index=False, engine='openpyxl')

        print(f"\n✓ Total messages collected: {len(all_messages)}")
        print(f"✓ Data saved:\n  XLSX → {xlsx_path}")

        print("\n" + "=" * 60)
        print("COLLECTION COMPLETE")
        print("=" * 60)


if __name__ == "__main__":
    main()
