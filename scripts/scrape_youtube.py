import os
import csv
import sys
try:
    from yt_dlp import YoutubeDL
except ImportError:
    print("[!] yt-dlp is not installed in the virtual environment. Please run: pip install yt-dlp")
    sys.exit(1)

# Reconfigure stdout to support unicode prints on Windows console without crashes
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Dynamic search queries to find active videos for various domains in Nepal
SEARCH_CONFIGS = [
    {
        "query": "ytsearch2:Redmi Note review in Nepali",
        "category": "Tech",
        "name_prefix": "Redmi Note Review"
    },
    {
        "query": "ytsearch2:iPhone review in Nepali",
        "category": "Tech",
        "name_prefix": "iPhone Review"
    },
    {
        "query": "ytsearch2:Pathao InDrive Nepal review",
        "category": "Ride-Sharing",
        "name_prefix": "Pathao vs InDrive"
    },
    {
        "query": "ytsearch2:Daraz Nepal online shopping review",
        "category": "E-Commerce",
        "name_prefix": "Daraz Shopping"
    },
    {
        "query": "ytsearch2:Worldlink internet review Nepal",
        "category": "ISP",
        "name_prefix": "Worldlink ISP Review"
    }
]

def scrape_nepali_comments(search_configs, output_file="data/raw/raw_comments.csv", max_comments_per_video=100):
    """
    Dynamically searches YouTube for relevant Nepali topics, extracts comments
    from the active search results, and saves them to a CSV file.
    Does not require a YouTube API key.
    """
    print(f"[*] Initializing dynamic YouTube search-scraper using yt-dlp backend...")
    
    # Ensure target directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    total_scraped = 0
    
    print(f"[*] Writing comments to {output_file}...")
    with open(output_file, mode='w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        # Write headers
        writer.writerow(["CommentID", "Author", "CommentText", "Likes", "PublishedTime", "SourceVideo", "Category"])
        
        for config in search_configs:
            query = config["query"]
            category = config["category"]
            prefix = config["name_prefix"]
            
            print(f"\n[*] Searching: '{query}' for Category: [{category}]")
            
            ydl_opts = {
                "getcomments": True,       # Enable comment extraction
                "skip_download": True,     # Do not download video
                "quiet": True,             # Suppress verbose output
                "no_warnings": True,
                "extractor_args": {
                    "youtube": {
                        "max_comments": [str(max_comments_per_video)],
                    },
                },
            }
            
            try:
                with YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(query, download=False)
                    entries = info.get("entries") or []
            except Exception as e:
                print(f"[-] Search failed for query '{query}': {e}. Skipping...")
                continue
                
            print(f"[+] Found {len(entries)} active video(s) for this query.")
            
            for v_idx, entry in enumerate(entries):
                title = entry.get("title") or f"{prefix} Video {v_idx+1}"
                comments = entry.get("comments") or entry.get("comment_entries") or []
                
                # Safeguard title encoding for print statements
                try:
                    print(f"    --> Scraping Video: '{title[:60]}...'")
                except UnicodeEncodeError:
                    safe_title = title.encode('ascii', 'ignore').decode('ascii')
                    print(f"    --> Scraping Video: '{safe_title[:60]}...'")
                
                video_count = 0
                for comment in comments:
                    writer.writerow([
                        comment.get('id'),
                        comment.get('author'),
                        comment.get('text'),
                        comment.get('like_count', 0),
                        comment.get('time_text') or comment.get('timestamp') or 'Unknown',
                        title,
                        category
                    ])
                    video_count += 1
                    total_scraped += 1
                    
                print(f"        [+] Collected {video_count} comments from this video.")
                
    print(f"\n[+] Dynamic batch scraping completed successfully!")
    print(f"[+] Total comments collected across all sources: {total_scraped}")
    print(f"[+] Dataset saved to: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    output_path = "data/raw/raw_comments.csv"
    
    if len(sys.argv) > 1:
        # Check if a custom URL or query was passed as command line argument
        custom_input = sys.argv[1]
        print(f"[*] Started with custom input: {custom_input}")
        
        # If it looks like a URL, treat it as a direct link, otherwise search for it
        if custom_input.startswith("http"):
            configs = [{
                "query": custom_input,
                "category": "Custom",
                "name_prefix": "Custom Video Link"
            }]
        else:
            configs = [{
                "query": f"ytsearch2:{custom_input}",
                "category": "Custom",
                "name_prefix": f"Search: {custom_input}"
            }]
        scrape_nepali_comments(configs, output_path, max_comments_per_video=200)
    else:
        # Run batch of diverse default search configurations
        print("[*] No custom input provided. Executing batch searches for multi-domain Nepali content...")
        scrape_nepali_comments(SEARCH_CONFIGS, output_path, max_comments_per_video=100)
