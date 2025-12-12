import pandas as pd
import re
from collections import Counter

# --- Configuration ---
# Define the paths to the labeled datasets
LABELED_FILES = {
    "Reddit": 'data/reddit_labeled.xlsx',
    # Assuming the YouTube labeled data follows a similar naming convention
    "YouTube": 'data/youtube_labeled.xlsx'
}

def analyze_confidence_scores(file_path, dataset_name):
    """
    Reads a labeled dataset, extracts confidence scores from 'Annotator notes',
    and returns a Counter of the confidence levels.
    """
    print(f"--- Analyzing {dataset_name} Data ---")
    
    # 1. Check if file exists
    if not pd.io.common.file_exists(file_path):
        print(f"⚠️ File not found at: {file_path}. Skipping analysis for {dataset_name}.")
        print("-" * 40)
        return

    try:
        # 2. Load the dataset
        df = pd.read_excel(file_path)
        print(f"Total rows loaded: {len(df)}")
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        print("-" * 40)
        return

    # 3. Extract confidence scores
    # Regex to find 'High', 'Medium', or 'Low' following 'Confidence: '
    confidence_pattern = re.compile(r"Confidence:\s*(High|Medium|Low)", re.IGNORECASE)
    
    all_confidences = []

    # Iterate through the 'Annotator notes' column
    # The data is stored as: "Confidence: <Level> | <Reasoning>"
    for notes in df['Annotator notes'].astype(str):
        match = confidence_pattern.search(notes)
        if match:
            # Group 1 captures 'High', 'Medium', or 'Low'
            confidence = match.group(1).capitalize()
            all_confidences.append(confidence)
        else:
            # Count unlabeled or unparsed entries separately
            all_confidences.append('Unlabeled/Unparsed')
            
    # 4. Count the frequencies
    confidence_counts = Counter(all_confidences)
    
    # 5. Print the summary report
    print("\n✅ Confidence Score Summary:")
    
    # Define expected order for clarity
    order = ['High', 'Medium', 'Low', 'Unlabeled/Unparsed']
    
    for level in order:
        count = confidence_counts.get(level, 0)
        print(f"  {level:<10}: {count:>8} ({count/len(df)*100:.2f}%)")

    print("-" * 40)


def main():
    """Iterate through all defined datasets and run the analysis."""
    for dataset_name, file_path in LABELED_FILES.items():
        analyze_confidence_scores(file_path, dataset_name)
    
    print("Confidence analysis complete for all datasets.")


if __name__ == "__main__":
    main()
