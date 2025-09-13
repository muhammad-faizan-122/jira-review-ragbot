import json
import os
import re
from datetime import datetime


def clean_text(text):
    """
    Cleans a text string by removing extra whitespace, quotes, and newlines.

    Args:
        text (str): The input string to clean.

    Returns:
        str or None: The cleaned string, or None if the input is not a string.
    """
    if not isinstance(text, str):
        return None  # Return None if input is not a string (e.g., None, int)
    cleaned_text = text.strip().strip("\"'")
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    return cleaned_text


def validate_and_prepare_review(review):
    """
    Validates a raw review, ensures all required keys are present, handles aliases,
    and filters out unnecessary keys.
    """
    required_keys = [
        "title",
        "author",
        "review_date",
        "rating",
        "pros",
        "cons",
        "review_detail",
    ]
    prepared_review = {}

    for key in required_keys:
        if key == "rating":
            prepared_review[key] = review.get("rating") or review.get("overall_rating")
        else:
            prepared_review[key] = review.get(key)

    return prepared_review


def standardize_date(date_string):
    """
    Parses a date string from known formats and returns it in "Month Year" format.
    """
    if not date_string or not isinstance(date_string, str):
        return None

    cleaned_string = date_string.replace("Reviewed ", "").strip()

    try:  # Format 1: "September 2025"
        date_obj = datetime.strptime(cleaned_string, "%B %Y")
        return date_obj.strftime("%B %Y")
    except ValueError:
        pass

    try:  # Format 2: "8/27/2025"
        date_obj = datetime.strptime(cleaned_string, "%m/%d/%Y")
        return date_obj.strftime("%B %Y")
    except ValueError:
        pass

    print(
        f"  - Warning: Could not parse date format for '{date_string}'. Setting to None."
    )
    return None


def standardize_review_values(prepared_review):
    """
    Standardizes the values (rating, date, text fields) of a pre-validated review.
    """
    updated_reviews = {}
    standardized_review = prepared_review.copy()

    # --- 1. Standardize Rating Value ---
    raw_rating = standardized_review.get("rating")
    if raw_rating:
        match = re.search(r"[\d.]+", str(raw_rating))
        updated_reviews["rating"] = float(match.group(0)) if match else None
    else:
        updated_reviews["rating"] = None

    # --- 2. Standardize Date Value ---
    raw_date = standardized_review.get("review_date")
    updated_reviews["review_date"] = standardize_date(raw_date)

    # --- 3. Clean Text Fields ---
    updated_reviews["review_detail"] = ""
    for key in ["title", "review_detail", "pros", "cons"]:
        # ignore empty
        if not standardized_review.get(key):
            continue
        updated_reviews[
            "review_detail"
        ] += f"{key}: {clean_text(standardized_review.get(key))}\n"
    # 4. add author name
    updated_reviews["author"] = standardized_review.get("author")

    return updated_reviews


def save_json(data, output_file):
    # --- Save the combined and standardized data ---
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"\n✅ Successfully processed and standardized {len(data)} reviews.")
        print(f"Data saved to '{output_file}'")
    except Exception as e:
        print(f"\n❌ Error saving the final JSON file: {e}")


def process_review_files(
    json_dir="atlassian_jira_reviews_data", output_file="standardized_reviews.json"
):
    """
    Main pipeline to read, validate, standardize, and save review data.
    """
    if not os.path.exists(json_dir):
        print(f"Error: Directory not found at '{json_dir}'")
        return

    all_final_reviews = []
    print(f"Reading JSON files from '{json_dir}'...")

    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(json_dir, filename)
            # print(f"Processing '{filename}'...")

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    reviews_data = json.load(f)

                for raw_review in reviews_data:
                    prepared_review = validate_and_prepare_review(raw_review)
                    final_review = standardize_review_values(prepared_review)
                    all_final_reviews.append(final_review)

            except json.JSONDecodeError:
                print(
                    f"  - Warning: Could not decode JSON from '{filename}'. Skipping."
                )
            except Exception as e:
                print(f"  - An unexpected error occurred with '{filename}': {e}")

    save_json(data=all_final_reviews, output_file=output_file)
    return all_final_reviews


JSON_DIRS = ["g2_jira_reviews_data", "atlassian_jira_reviews_data"]
all_reviews = []
for JSON_DIR in JSON_DIRS:
    output_json_path = f"scrapped_merged_data/{JSON_DIR}.json"
    # --- Run the main pipeline function ---
    clean_reviews = process_review_files(
        json_dir=JSON_DIR, output_file=output_json_path
    )

    all_reviews.extend(clean_reviews)
save_json(data=all_reviews, output_file="scrapped_merged_data/all_reviews.json")
