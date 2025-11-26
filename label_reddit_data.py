"""
Reddit Data Labeling Script using Gemini API
Uses batch processing with 5 rows per request and 3 API loops
Respects rate limits: RPM 15, RPD 1000 (Gemini 2.5 Flash Lite)
"""

import pandas as pd
import google.generativeai as genai
import time
import os
from datetime import datetime
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Configuration
GEMINI_API_KEY_1 = os.getenv('GEMINI_API_KEY_1')
GEMINI_API_KEY_2 = os.getenv('GEMINI_API_KEY_2')
GEMINI_API_KEY_3 = os.getenv('GEMINI_API_KEY_3')
GEMINI_API_KEY_4 = os.getenv('GEMINI_API_KEY_4')
GEMINI_API_KEY_5 = os.getenv('GEMINI_API_KEY_5')
API_KEYS = [GEMINI_API_KEY_1, GEMINI_API_KEY_2, GEMINI_API_KEY_3, GEMINI_API_KEY_4, GEMINI_API_KEY_5]

# Rate limiting configuration
RPM = 15  # Requests per minute
RPD = 1000  # Requests per day
BATCH_SIZE = 10  # Number of rows to process in one request
DELAY_BETWEEN_REQUESTS = 2  # Delay in seconds (optimized for 10 API keys)

# File paths
INPUT_FILE = 'data/reddit.xlsx'
OUTPUT_FILE = 'data/reddit_labeled.xlsx'
PROGRESS_FILE = 'reddit_progress.json'


def load_annotation_guidelines():
    """Load annotation guidelines from PDF or text file"""
    guidelines = """
    ANNOTATION GUIDELINES FOR ISRAEL-GAZA WAR DISCOURSE
    
    Objective: Label each datapoint with its corresponding stance towards the Israel-Gaza conflict.
    
    STANCE LABELS:
    
    P = Supports Palestine
    - Advocates for Palestinian rights, interests, or perspectives
    - Supports Palestinian statehood, sovereignty, self-determination, independence, and equality
    - Criticizes Israel's actions or policies towards Palestinians
    - Examples:
      * "Hamas wanted to negotiate the week of the 7th. USA/Israel said no."
      * "Not all Palestinians support Hamas. However, being able to express it leads to death in Gaza."
      * "It is more than just Israel. It is the imperialist countries, too, including the United States."
    
    I = Supports Israel
    - Supportive of Israel's interests, security, and rights
    - Backs Israel's sovereignty, territorial integrity, and protection of citizens
    - Supports Israel's right to defend itself and ensure survival
    - Examples:
      * "It's the ultimate gaslighting to blame Israel for people deciding to be terrorists."
      * "Israel doesn't occupy Gaza, for the past 18 years. They pulled settlers and military out in 2005."
      * "Thank you Israel. Taking the garbage out now so we don't have to in 5 years."
    
    N = Neutral/Unclear Stance
    - Impartial or ambiguous viewpoint
    - No definitive position favoring either Palestinian or Israeli side
    - Lacks sufficient information or intentionally avoids partisanship
    - Presents balanced views or asks questions without taking sides
    - Examples:
      * "There's a pretty large difference between engineers making nuclear weapons vs. random Israeli civilians."
      * "All I ask is for my legitimate questions to be legitimately answered."
    
    IMPORTANT INSTRUCTIONS:
    - Evaluate based on context, tone, and content
    - Consider nuances and subtleties in language
    - Avoid assumptions or bias
    - Some datapoints may be incomplete (part of ongoing conversation)
    
    Provide labels in JSON format with fields: "Label" (must be P, I, or N), "Confidence", "Reasoning"
    """
    return guidelines


def create_prompt_for_batch(batch_df, guidelines):
    """Create a prompt for a batch of Reddit posts/comments"""
    prompt = f"""You are an expert annotator analyzing discourse about the Israel-Hamas war on Reddit.

{guidelines}

Analyze the following {len(batch_df)} Reddit posts/comments and provide labels for each.

For each item, consider:
- The post title and content
- The comment text
- The subreddit context
- Any engagement metrics (score, controversiality)

Data to analyze:
"""
    
    for idx, row in batch_df.iterrows():
        prompt += f"\n\n--- Item {idx} ---\n"
        prompt += f"Subreddit: {row['subreddit']}\n"
        prompt += f"Post Title: {row['post_title']}\n"
        prompt += f"Post Content: {row['post_self_text'][:500] if pd.notna(row['post_self_text']) else 'N/A'}\n"
        prompt += f"Comment: {row['self_text'][:500] if pd.notna(row['self_text']) else 'N/A'}\n"
        prompt += f"Score: {row['score']}, Controversiality: {row['controversiality']}\n"
    
    prompt += f"""

Respond with a JSON array containing exactly {len(batch_df)} objects, one for each item analyzed above.
Each object should have this structure:
{{
    "index": <the item index>,
    "Label": "<P or I or N>",
    "Confidence": "<High/Medium/Low>",
    "Reasoning": "<brief explanation of why this label was chosen>"
}}

IMPORTANT: 
- Label must be EXACTLY one of: P, I, or N
- P = Supports Palestine
- I = Supports Israel  
- N = Neutral/Unclear Stance

Respond ONLY with the JSON array, no additional text."""
    
    return prompt


def parse_gemini_response(response_text):
    """Parse Gemini API response to extract labels"""
    try:
        # Try to extract JSON from response
        # Sometimes the model wraps JSON in markdown code blocks
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_text.strip()
        
        # Try parsing with strict=False to handle escape issues
        labels = json.loads(json_str, strict=False)
        return labels
    except json.JSONDecodeError as e:
        # If standard parsing fails, try manual cleanup
        try:
            import re
            # Fix common escape issues by replacing problematic backslashes
            json_str_cleaned = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_str)
            labels = json.loads(json_str_cleaned, strict=False)
            return labels
        except Exception as e2:
            print(f"Error parsing response after cleanup: {e2}")
            print(f"Response text: {response_text[:500]}")
            return None
    except Exception as e:
        print(f"Error parsing response: {e}")
        print(f"Response text: {response_text[:500]}")
        return None


def is_rate_limit_error(error):
    """Check if error is a rate limit error"""
    error_str = str(error).lower()
    return any([
        '429' in error_str,
        'resource_exhausted' in error_str,
        'rate limit' in error_str,
        'quota' in error_str
    ])


def label_batch_with_gemini(batch_df, api_keys, current_key_idx, guidelines):
    """Label a batch of rows using Gemini API with infinite retry logic"""
    attempts = 0
    backoff_time = 2
    keys_tried_in_round = 0
    
    while True:  # Keep trying until success
        try:
            # Select API key
            api_key = api_keys[current_key_idx % len(api_keys)]
            
            # Configure API
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash-lite')
            
            # Create prompt
            prompt = create_prompt_for_batch(batch_df, guidelines)
            
            # Generate response
            response = model.generate_content(prompt)
            
            # Check if response is valid
            if not response or not hasattr(response, 'text'):
                print(f"⚠️  Invalid response object, retrying...")
                attempts += 1
                time.sleep(backoff_time)
                backoff_time = min(backoff_time * 1.5, 60)
                continue
            
            # Parse response
            labels = parse_gemini_response(response.text)
            
            if labels:
                return labels, current_key_idx
            else:
                # Parsing failed, retry with backoff
                print(f"⚠️  Failed to parse response, retrying...")
                attempts += 1
                time.sleep(backoff_time)
                backoff_time = min(backoff_time * 1.5, 60)  # Cap at 60 seconds
                
        except Exception as e:
            attempts += 1
            error_msg = str(e)
            
            # Skip batch if subscriptable error persists after 3 attempts
            if 'subscriptable' in error_msg and attempts >= 3:
                print(f"⚠️  Skipping batch due to persistent error: {error_msg}")
                # Return empty list to skip this batch without labeling
                return [], current_key_idx
            
            if is_rate_limit_error(e):
                print(f"⚠️  Rate limit hit on API key {current_key_idx % len(api_keys) + 1}")
                keys_tried_in_round += 1
                
                # Try next API key
                current_key_idx += 1
                print(f"🔄 Switching to API key {current_key_idx % len(api_keys) + 1}")
                
                # If we've cycled through all keys, wait before trying again
                if keys_tried_in_round >= len(api_keys):
                    print(f"❌ All {len(api_keys)} API keys exhausted. Waiting {backoff_time}s before retry...")
                    time.sleep(backoff_time)
                    backoff_time = min(backoff_time * 2, 300)  # Cap at 5 minutes
                    keys_tried_in_round = 0  # Reset counter
                else:
                    time.sleep(1)  # Brief pause before trying next key
            else:
                print(f"Error calling Gemini API (attempt {attempts}): {e}")
                time.sleep(backoff_time)
                backoff_time = min(backoff_time * 1.5, 60)  # Cap at 60 seconds


def save_progress(processed_count, total_count):
    """Save progress to file"""
    progress = {
        'processed_count': processed_count,
        'total_count': total_count,
        'last_updated': datetime.now().isoformat()
    }
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f)


def load_progress():
    """Load progress from file"""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {'processed_count': 0}


def main():
    """Main function to process Reddit data"""
    print("=" * 80)
    print("Reddit Data Labeling Script")
    print("=" * 80)
    
    # Check API keys - filter for valid strings only
    valid_keys = [key for key in API_KEYS if key and isinstance(key, str) and key.startswith('AIza')]
    if not valid_keys:
        print("ERROR: No valid API keys found. Please check your .env file")
        print("Loaded keys:", [type(key).__name__ for key in API_KEYS])
        return
    
    print(f"Found {len(valid_keys)} valid API key(s)")
    
    # Load data
    print(f"\nLoading data from {INPUT_FILE}...")
    df = pd.read_excel(INPUT_FILE)
    print(f"Loaded {len(df)} rows")
    
    # Load annotation guidelines
    guidelines = load_annotation_guidelines()
    
    # Load progress
    progress = load_progress()
    start_idx = progress.get('processed_count', 0)
    
    if start_idx > 0:
        print(f"\nResuming from row {start_idx}")
    
    # Process data in batches
    total_rows = len(df)
    current_api_idx = 0
    requests_today = 0
    requests_this_minute = 0
    minute_start_time = time.time()
    
    print(f"\nProcessing {total_rows} rows in batches of {BATCH_SIZE}")
    print(f"Rate limits: {RPM} requests/minute, {RPD} requests/day")
    print(f"Delay between requests: {DELAY_BETWEEN_REQUESTS:.1f} seconds")
    print("-" * 80)
    
    for batch_start in range(start_idx, total_rows, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_rows)
        batch_df = df.iloc[batch_start:batch_end]
        
        print(f"\nProcessing rows {batch_start} to {batch_end-1} ({len(batch_df)} rows)...")
        
        # Check daily limit
        if requests_today >= RPD:
            print(f"Reached daily limit of {RPD} requests. Stopping.")
            break
        
        # Check minute limit and wait if necessary
        current_time = time.time()
        if current_time - minute_start_time < 60:
            if requests_this_minute >= RPM:
                wait_time = 60 - (current_time - minute_start_time)
                print(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                requests_this_minute = 0
                minute_start_time = time.time()
        else:
            requests_this_minute = 0
            minute_start_time = time.time()
        
        # Use API keys in rotation with retry logic
        print(f"Using API key {current_api_idx % len(valid_keys) + 1}")
        
        # Label batch (returns labels and potentially updated key index)
        labels, current_api_idx = label_batch_with_gemini(
            batch_df, valid_keys, current_api_idx, guidelines
        )
        
        # Apply labels to dataframe
        for label_data in labels:
            try:
                idx = label_data['index']
                label = label_data.get('Label', 'N')
                # Validate label
                if label not in ['P', 'I', 'N']:
                    print(f"Warning: Invalid label '{label}' for index {idx}, defaulting to 'N'")
                    label = 'N'
                df.at[idx, 'Label'] = label
                df.at[idx, 'Annotator notes'] = f"Confidence: {label_data.get('Confidence', '')} | {label_data.get('Reasoning', '')}"
            except Exception as e:
                print(f"Error applying label for index {idx}: {e}")
        
        print(f"✓ Successfully labeled {len(labels)} items")
        
        # Update counters
        requests_today += 1
        requests_this_minute += 1
        current_api_idx += 1
        
        # Save progress
        save_progress(batch_end, total_rows)
        
        # Save intermediate results
        if (batch_end % 50) == 0 or batch_end == total_rows:
            df.to_excel(OUTPUT_FILE, index=False)
            print(f"💾 Saved progress to {OUTPUT_FILE}")
        
        # Wait before next request
        if batch_end < total_rows:
            print(f"Waiting {DELAY_BETWEEN_REQUESTS:.1f} seconds before next request...")
            time.sleep(DELAY_BETWEEN_REQUESTS)
    
    # Save final results
    print("\n" + "=" * 80)
    print("Processing complete!")
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"Final results saved to {OUTPUT_FILE}")
    print(f"Total requests made: {requests_today}")
    print("=" * 80)


if __name__ == "__main__":
    main()
