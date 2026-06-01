import sys
import os
import csv
import json

# Add the repo to Python path
sys.path.append(os.path.abspath('yt-bulk-subtitles-downloader'))

try:
    import ytbsd
except Exception as e:
    print(f"Failed to import ytbsd: {e}")

def main():
    print("YTBSD successfully imported.")

if __name__ == '__main__':
    main()
