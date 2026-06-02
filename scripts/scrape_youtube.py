import os
import csv
import sys
try:
    from yt_dlp import YoutubeDL
except ImportError:
    print("[!] yt-dlp is not installed in the virtual environment. Please run: pip install yt-dlp")
    sys.exit(1)

# Pre-defined list of diverse Nepali video reviews and discussions across different categories
DEFAULT_VIDEOS = [
    {
        "url": "https://www.youtube.com/watch?v=F05d0Bnvkac",
        "category": "Tech",
        "name": "Redmi Note 13 Review"
    },
    {
        "url": "https://www.youtube.com/watch?v=Jm3U-H5Wf50",
        "category": "Tech",
        "name": "iPhone 15 Pro Nepali Review"
    },
    {
        "url": "https://www.youtube.com/watch?v=N_8a39o1-0Y",
        "category": "Ride-Sharing",
        "name": "Pathao vs InDrive Nepal"
    },
    {
        "url": "https://www.youtube.com/watch?v=1uR1o7LuhbU",
        "category": "E-Commerce",
        "name": "Daraz Nepal Shopping Experience"
    },
    {
        "url": "https://www.youtube.com/watch?v=6K_nS1WjEms",
        "category": "ISP",
        "name": "Worldlink Internet Service Review"
    }
]

def scrape_nepali_comments(videos, output_file="data/raw/raw_comments.csv", max_comments_per_video=200):
    """
    Scrapes comments from a list of YouTube video URLs using yt-dlp and saves them to a single CSV file.
    Categorizes reviews to support multi-domain analysis.
    Does not require a YouTube API key.
    """
    print(f"[*] Initializing multi-domain YouTube downloader using yt-dlp backend...")
    
    # Ensure target directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    total_scraped = 0
    
    print(f"[*] Writing comments to {output_file}...")
    with open(output_file, mode='w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        # Write headers including metadata for source and category
        writer.writerow(["CommentID", "Author", "CommentText", "Likes", "PublishedTime", "SourceVideo", "Category"])
        
        for video_info in videos:
            url = video_info["url"]
            category = video_info["category"]
            name = video_info["name"]
            
            print(f"\n[*] Scraping Category: [{category}] | Name: '{name}'")
            print(f"    URL: {url}")
            
            # Configure yt-dlp extraction options
            ydl_opts = {
                "getcomments": True,       # Enable comment extraction
                "skip_download": True,     # Do not download the video itself
                "quiet": True,             # Suppress verbose log printouts
                "no_warnings": True,
                "extractor_args": {
                    "youtube": {
                        "max_comments": [str(max_comments_per_video)], # Limit fetched comments
                    },
                },
            }
            
            try:
                with YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    comments = info.get("comments") or info.get("comment_entries") or []
            except Exception as e:
                print(f"[-] Error extracting comments for video '{name}': {e}")
                print("    (Note: This could be due to country/geo-restrictions on the video or comments being disabled). Skipping...")
                continue
                
            video_count = 0
            for comment in comments:
                writer.writerow([
                    comment.get('id'),
                    comment.get('author'),
                    comment.get('text'),
                    comment.get('like_count', 0),
                    comment.get('time_text') or comment.get('timestamp') or 'Unknown',
                    name,
                    category
                ])
                video_count += 1
                total_scraped += 1
                
                if video_count % 50 == 0:
                    print(f"    [+] Scraped {video_count} comments...")
                    
            print(f"[+] Completed: Collected {video_count} comments from this video.")
            
    print(f"\n[+] Batch scraping completed successfully!")
    print(f"[+] Total comments collected across all sources: {total_scraped}")
    print(f"[+] Dataset saved to: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    output_path = "data/raw/raw_comments.csv"
    
    if len(sys.argv) > 1:
        # Check if a custom URL was passed as command line argument
        custom_url = sys.argv[1]
        print(f"[*] Started with custom URL: {custom_url}")
        videos_to_scrape = [{
            "url": custom_url,
            "category": "Custom",
            "name": "Custom Video Link"
        }]
        scrape_nepali_comments(videos_to_scrape, output_path, max_comments_per_video=300)
    else:
        # Run batch of diverse default videos
        print("[*] No custom URL provided. Starting batch scraping of multi-domain Nepali content...")
        scrape_nepali_comments(DEFAULT_VIDEOS, output_path, max_comments_per_video=200)
