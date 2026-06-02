import os
import csv
import sys
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR

def scrape_nepali_comments(video_url, output_file="data/raw_comments.csv", max_comments=500):
    """
    Scrapes comments from a specified YouTube video URL and saves them to a CSV file.
    Does not require a YouTube API key.
    """
    print(f"[*] Initializing downloader for: {video_url}")
    downloader = YoutubeCommentDownloader()
    
    # Ensure target directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        # Retrieve comments generator
        comments_gen = downloader.get_comments_from_url(video_url, sort_by=SORT_BY_POPULAR)
    except Exception as e:
        print(f"[-] Error accessing video URL: {e}")
        return
        
    count = 0
    
    print(f"[*] Scraping and writing up to {max_comments} comments to {output_file}...")
    with open(output_file, mode='w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        # Write headers
        writer.writerow(["CommentID", "Author", "CommentText", "Likes", "PublishedTime"])
        
        for comment in comments_gen:
            if count >= max_comments:
                break
                
            writer.writerow([
                comment.get('cid'),
                comment.get('author'),
                comment.get('text'),
                comment.get('likes'),
                comment.get('time')
            ])
            count += 1
            if count % 100 == 0:
                print(f"[+] Scraped {count} comments...")
                
    print(f"[+] Scraping completed successfully! Total comments collected: {count}")
    print(f"[+] Saved to: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    # Example video: A popular Nepali tech review video (e.g., GadgetByte review)
    # You can swap this out with any YouTube URL!
    default_video = "https://www.youtube.com/watch?v=F05d0Bnvkac" # GadgetByte Redmi Note 13 review or similar
    
    # Check if a custom URL was passed as command line argument
    url = sys.argv[1] if len(sys.argv) > 1 else default_video
    
    scrape_nepali_comments(url, max_comments=300)
