import csv
import Levenshtein
import pandas as pd
from urllib.request import urlretrieve

########################## Part 1: Load DiffusionDB ############################

# Download the parquet table
table_url = 'https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/metadata.parquet'
urlretrieve(table_url, 'metadata.parquet')

# Read the table using Pandas
raw_df = pd.read_parquet('metadata.parquet')

# Keep top 200K prompts
prompts_raw = raw_df['prompt'][0:14000000]

del raw_df  # Free memory

########################## Part 2: Filter by UI Keywords ########################

# Extended keyword list
ui_keywords = [
    "mobile UI", "app interface", "UI design", "UX design",
    "responsive layout", "app design", "mobile layout",
    "mobile screen", "UI prototype", "mobile wireframe",
    "interface design", "dashboard UI", "screen design",

]

# Broad keyword filtering
def broad_filter_ui_prompts(prompts, keywords):
    keywords_split = set(word.lower() for keyword in keywords for word in keyword.split())
    ui_related_prompts = []
    for prompt in prompts:
        if any(word in prompt.lower() for word in keywords_split):
            ui_related_prompts.append(prompt)
    return ui_related_prompts

# Apply broad filtering
prompts_filtered = broad_filter_ui_prompts(prompts_raw, ui_keywords)

print(f"Prompts after keyword filtering: {len(prompts_filtered)}")  # Debugging

########################## Part 3: Remove Similar Prompts ######################

# Sequential function to remove similar strings based on Levenshtein distance
def remove_similar_strings(strings, threshold=10, max_unique=1000):
    unique_strings = []

    def is_unique(s, candidates):
        """
        Check if string 's' is unique among a list of candidates.
        """
        for candidate in candidates:
            # Exit early if distance exceeds threshold
            if Levenshtein.distance(s, candidate) <= threshold:
                return False
        return True

    for i, string in enumerate(strings):
        if len(unique_strings) >= max_unique:
            break  # Stop once we reach the maximum unique prompts

        # Compare only against unique strings so far
        if is_unique(string, unique_strings):
            unique_strings.append(string)

        # Log progress every 1000 strings
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(strings)} strings. Unique count: {len(unique_strings)}")

    return unique_strings

# Apply sequential deduplication and stop at 1000 unique prompts
prompts_unique = remove_similar_strings(prompts_filtered, threshold=5, max_unique=10000)

print(f"Prompts after removing similar ones: {len(prompts_unique)}")  # Debugging

########################## Part 4: Save to CSV #################################

# Save the unique UI prompts to a CSV file
csv_file_name = "prompts_mobile_ui.csv"

with open(csv_file_name, mode="w", newline="", encoding="utf-8") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["prompt example"])
    for string in prompts_unique:  # Saving the unique prompts
        csv_writer.writerow([string])

print(f"Final CSV file saved: {csv_file_name}")
