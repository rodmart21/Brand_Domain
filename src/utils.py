import pandas as pd
import re
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from fuzzywuzzy import fuzz
from rapidfuzz import fuzz as rfuzz
import openai
import time
import json
from typing import List, Dict
import os
from serpapi import GoogleSearch
from googleapiclient.discovery import build


config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
with open(config_path, "r") as file:
    config = json.load(file)

openai.api_key = config["openai_api_key"]
serp_api_key = config["serp_api_key"]
openai_api_key = config["openai_api_key"]
google_api_key = config["google_custom_search"]["api_key"]
search_engine_id = config["google_custom_search"]["search_engine_id"]


def preprocess_name(name):
    """Cleans manufacturer names by removing punctuation, common suffixes, and extra spaces."""
    if pd.isna(name):
        return ""

    name = name.lower()  # Convert to lowercase
    name = re.sub(r'[^\w\s]', ' ', name)  # Remove punctuation
    # name = re.sub(r'\b(inc|ltd|llc|corp|gmbh|co|group|sa|ag)\b', '', name)  # Remove legal suffixes
    name = re.sub(r'\s+', ' ', name).strip()  # Remove extra spaces
    return name

def jaccard_similarity(name1, name2):
    """Computes Jaccard similarity between two manufacturer names based on word sets."""
    set1, set2 = set(name1.split()), set(name2.split())
    if not set1 or not set2:
        return 0
    return len(set1 & set2) / len(set1 | set2)

def levenshtein_ratio(name1, name2):
    """Computes Levenshtein similarity ratio (normalized edit distance)."""
    return fuzz.ratio(name1, name2) / 100  # Normalize to [0,1] scale

def hybrid_similarity(name1, name2):
    """Combines Jaccard similarity and Levenshtein ratio for a robust similarity score."""
    name1, name2 = preprocess_name(name1), preprocess_name(name2)
    
    jaccard_score = jaccard_similarity(name1, name2)
    levenshtein_score = levenshtein_ratio(name1, name2)
    
    # Weighted combination
    return 0.4 * jaccard_score + 0.6 * levenshtein_score


def find_best_match(name, candidates):
    """Finds the best match from a list using hybrid similarity."""
    best_match, best_score = None, 0
    
    for candidate in candidates:
        score = hybrid_similarity(name, candidate)
        if score > best_score:
            best_match, best_score = candidate, score
    
    return best_match, best_score

def calculate_similarity_scores(df, col1, col2):
    """
    Computes similarity scores between two columns in a DataFrame.

    Parameters:
    - df: pandas DataFrame
    - col1: First column name containing manufacturer names
    - col2: Second column name containing manufacturer names

    Returns:
    - DataFrame with an additional column 'Similarity_Score'
    """
    df = df.copy()
    df["Similarity_Score"] = df.apply(lambda row: hybrid_similarity(row[col1], row[col2]), axis=1)
    return df

# Example usage:
# df = pd.DataFrame({"Manufacturer_1": ["A.R. WILFLEY & SONS", "SKF Bearings"], "Manufacturer_2": ["AR Wilfley and Sons", "SKF"]})
# df_with_scores = calculate_similarity_scores(df, "Manufacturer_1", "Manufacturer_2")
# print(df_with_scores)

# Google Search


def google_search(query):
    service = build("customsearch", "v1", developerKey=google_api_key)
    result = service.cse().list(q=query, cx=search_engine_id).execute()
    
    if "items" in result:
        return result["items"][0]["title"], result["items"][0]["link"]
    return "No results found", ""

def search_company(company_name):
    params = {
        "engine": "google",
        "q": f"{company_name} company",
        "api_key": serp_api_key
    }
    
    search = GoogleSearch(params)
    results = search.get_dict()

    if "organic_results" in results and results["organic_results"]:
        return "Exists"
    else:
        return "Not Found"


def google_search_df(query):
    try:
        service = build("customsearch", "v1", developerKey=google_api_key)
        result = service.cse().list(q=query, cx=search_engine_id).execute()
        
        if "items" in result:
            return "Yes", result["items"][0]["link"]  # Exists = Yes, Website = First result
    except Exception as e:
        print(f"Error searching for {query}: {str(e)}")
    
    return "No", ""  # If no results, return No & empty website





# Prompting GPT

def standardize_manufacturers_df(df, input_col, output_col):
    """
    Processes a DataFrame column containing manufacturer names, standardizes each name using OpenAI, 
    and writes the results into another column.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        input_col (str): Name of the input column containing manufacturer names.
        output_col (str): Name of the output column to store the standardized names.

    Returns:
        pd.DataFrame: The modified DataFrame with the standardized names in the output column.
    """
    # Ensure the output column exists
    df[output_col] = None

    # Loop through each row
    for idx, name in df[input_col].items():
        prompt = f"""
        Extract only the **core company name** from the given manufacturer name and return it in a **clean and simplified format**.
        
        - Remove any additional words related to **divisions, subsidiaries, or products** such as "Filtersysteme", "Industriefiltration", "Bearings", etc.
        - Remove **legal forms** such as 'AG', 'Inc.', 'Ltd.', 'Corp.', 'Corporation', 'GmbH', 'Co.', 'Group', etc.
        - If the company name includes extra descriptors like "Automotive", "Filtration", "AKO", or similar, keep only the core brand name.
        - Keep only the **most essential name** that represents the entire company.

        **Example Output (JSON format):**
        {{
            "Original Name": "{name}",
            "Classification": "Company / Brand / Alias",
            "Standardized Name": "Core Company Name"
        }}

        **Example Conversions:**
        - "Mahle Filtersysteme GmbH" → "Mahle"
        - "Mahle AKO GmbH" → "Mahle"
        - "Mahle Industriefiltration" → "Mahle"
        - "SKF Bearings" → "SKF"
        - "Siemens AG" → "Siemens"
        
        Now process: "{name}"
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in company name standardization."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )

            result_text = response["choices"][0]["message"]["content"]
            result_dict = eval(result_text)  # Convert JSON-like string to dictionary

            # Write result to output column
            df.at[idx, output_col] = result_dict.get("Standardized Name", None)
            
            # Avoid hitting API rate limits
            time.sleep(1)

        except Exception as e:
            df.at[idx, output_col] = f"Error: {str(e)}"

    return df


def standardize_manufacturers_df_new(df: pd.DataFrame, input_col: str, output_col: str) -> pd.DataFrame:
    """
    Processes a DataFrame column containing manufacturer names, maps them to their parent company 
    or widely recognized global brand using OpenAI, and writes the results into another column.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        input_col (str): Name of the input column containing manufacturer names.
        output_col (str): Name of the output column to store standardized names.

    Returns:
        pd.DataFrame: The modified DataFrame with standardized names in the output column.
    """
    # Ensure the output column exists
    df[output_col] = None

    # Loop through each row
    for idx, name in df[input_col].items():
        prompt = f"""
        Map the given manufacturer name to its **parent company** or the **widely recognized global brand**.
        If the given name is a subsidiary, product line, or acquired brand, return the **most well-known company name**.
        If no parent company exists, return the original name.
        
        **Example Conversions:**
        - "Speedglas" → "3M"
        - "Adaptaflex" → "ABB"
        - "Mahle Filtersysteme" → "Mahle"
        - "KUKA Robotics" → "KUKA"
        - "Bosch Rexroth" → "Bosch"
        - "Hella KGaA Hueck & Co" → "Hella"
        
        **Example Output (JSON format):**
        {{
            "Original Name": "{name}",
            "Standardized Name": "Parent or Recognized Brand"
        }}
        
        Now process: "{name}"
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in company name standardization and corporate parentage."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )

            result_text = response["choices"][0]["message"]["content"]
            result_dict = json.loads(result_text)  # Convert JSON string to dictionary

            # Write the standardized name to the output column
            df.at[idx, output_col] = result_dict.get("Standardized Name", None)

            # Avoid hitting API rate limits
            time.sleep(1)

        except Exception as e:
            df.at[idx, output_col] = f"Error: {str(e)}"

    return df


def standardize_manufacturer_name(name: str) -> str:
    """
    Standardizes an individual manufacturer name using OpenAI.

    Parameters:
        name (str): Manufacturer name to standardize.

    Returns:
        str: Standardized name or an error message.
    """
    prompt = f"""
    Map the given manufacturer name to its **parent company** or the **widely recognized global brand**.
    If the given name is a subsidiary, product line, or acquired brand, return the **most well-known company name**.
    If no parent company exists, return the original name.

    **Example Conversions:**
    - "Speedglas" → "3M"
    - "Adaptaflex" → "ABB"
    - "Mahle Filtersysteme" → "Mahle"
    - "KUKA Robotics" → "KUKA"
    - "Bosch Rexroth" → "Bosch"
    - "Hella KGaA Hueck & Co" → "Hella"

    **Example Output (JSON format):**
    {{
        "Original Name": "{name}",
        "Standardized Name": "Parent or Recognized Brand"
    }}

    Now process: "{name}"
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in company name standardization and corporate parentage."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        result_text = response["choices"][0]["message"]["content"]
        result_dict = json.loads(result_text)  # Convert JSON string to dictionary

        return result_dict.get("Standardized Name", name)  # Return name if standardization fails

    except Exception as e:
        return f"Error: {str(e)}"


# Possible matchings:

def word_based_match(processed_name, standardized_names):
    processed_words = set(re.findall(r'\w+', processed_name.lower()))  # Extract words and lowercase them
    for s in standardized_names:
        standard_words = set(re.findall(r'\w+', s.lower()))
        if processed_words & standard_words:  # Check if there's any word overlap
            return s
    return None


# Standarization in batches
def batch_classify_and_save(unique_names: List[str], output_file: str, batch_size: int = 50):
    """
    Processes a list of unique manufacturer names in batches, standardizes each using OpenAI, 
    and saves the results incrementally into a CSV file.

    Parameters:
        unique_names (List[str]): List of unique manufacturer names.
        output_file (str): Path to save the output CSV file.
        batch_size (int, optional): Number of names to process per batch. Default is 50.

    Returns:
        pd.DataFrame: DataFrame containing all standardized results.
    """
    # Initialize existing_df as an empty DataFrame
    existing_df = pd.DataFrame()

    for i in range(0, len(unique_names), batch_size):
        batch = unique_names[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1} of {len(unique_names) // batch_size + 1}...")

        # Process each name individually
        batch_results = [{"Original Name": name, "Standardized Name": standardize_manufacturer_name(name)} for name in batch]

        # Convert batch results to a DataFrame
        batch_df = pd.DataFrame(batch_results)

        # Append to existing results and save
        if not existing_df.empty:
            combined_df = pd.concat([existing_df, batch_df]).drop_duplicates()
        else:
            combined_df = batch_df

        # Update existing_df to store the current progress
        existing_df = combined_df

        # Save progress incrementally
        combined_df.to_csv(output_file, index=False)
        print(f"Saved progress: {output_file}")

    return existing_df
