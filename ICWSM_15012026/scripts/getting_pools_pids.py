import json
import csv
import os
import glob

def extract_geocoding_data(folder_path, output_file):
    """
    Extract place_id, latitude, and longitude from JSON files
    
    Args:
        folder_path: Path to folder containing JSON files
        output_file: Name of output CSV file
    """
    
    # Get all JSON files in the folder
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    print(f"Found {len(json_files)} JSON files.")
    if not json_files:
        print("No JSON files found in the specified folder.")
        return
    
    extracted_data = []
    
    # Process each JSON file
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
                # Extract the required fields
                place_id = None
                latitude = None
                longitude = None
                
                # Extract latitude and longitude from input section
                if 'input' in data:
                    input_data = data['input']
                    latitude = input_data.get('latitude')
                    longitude = input_data.get('longitude')
                
                # Extract place_id from reverse_geocoding results
                if 'reverse_geocoding' in data:
                    reverse_geocoding = data['reverse_geocoding']
                    if 'results' in reverse_geocoding and isinstance(reverse_geocoding['results'], list) and len(reverse_geocoding['results']) > 0:
                        # Assuming we want the first result's place_id
                        first_result = reverse_geocoding['results'][0]
                        place_id = first_result.get('place_id')
                if 'place_id' in data:
                    place_id = data['place_id']
                # Only add if we found at least one of the fields
                if place_id or latitude or longitude:
                    extracted_data.append({
                        'place_id': place_id,
                        'latitude': latitude,
                        'longitude': longitude,
                        'source_file': os.path.basename(json_file)
                    })
                else:
                    print(f"No relevant data found in {json_file}")
                    
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    if not extracted_data:
        print("No data extracted from any files.")
        return
    
    # Write to CSV
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['place_id', 'latitude', 'longitude', 'source_file']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(extracted_data)
        
        print(f"Successfully extracted {len(extracted_data)} records to {output_file}")
        
    except Exception as e:
        print(f"Error writing CSV: {e}")

# # Alternative version if you want to handle multiple results per file
# def extract_all_geocoding_results(folder_path, output_file= output_file):
#     """
#     Extract all results from each JSON file (multiple rows per file if multiple results)
#     """
    
#     json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
#     if not json_files:
#         print("No JSON files found in the specified folder.")
#         return
    
#     extracted_data = []
    
#     for json_file in json_files:
#         try:
#             with open(json_file, 'r', encoding='utf-8') as file:
#                 data = json.load(file)
                
#                 # Extract latitude and longitude (same for all results in this file)
#                 latitude = None
#                 longitude = None
                
#                 if 'input' in data:
#                     input_data = data['input']
#                     latitude = input_data.get('latitude')
#                     longitude = input_data.get('longitude')
                
#                 # Extract all place_ids from results
#                 if 'reverse_geocoding' in data:
#                     reverse_geocoding = data['reverse_geocoding']
#                     if 'results' in reverse_geocoding and isinstance(reverse_geocoding['results'], list):
#                         for result in reverse_geocoding['results']:
#                             place_id = result.get('place_id')
#                             if place_id:  # Only add if place_id exists
#                                 extracted_data.append({
#                                     'place_id': place_id,
#                                     'latitude': latitude,
#                                     'longitude': longitude,
#                                     'source_file': os.path.basename(json_file)
#                                 })
                
#                 # If no results were found but we have coordinates, still add one row
#                 if not any(item['source_file'] == os.path.basename(json_file) for item in extracted_data) and (latitude or longitude):
#                     extracted_data.append({
#                         'place_id': None,
#                         'latitude': latitude,
#                         'longitude': longitude,
#                         'source_file': os.path.basename(json_file)
#                     })
                    
#         except Exception as e:
#             print(f"Error processing {json_file}: {e}")
    
#     if not extracted_data:
#         print("No data extracted from any files.")
#         return
    
#     # Write to CSV
#     try:
#         with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
#             fieldnames = ['place_id', 'latitude', 'longitude', 'source_file']
#             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#             writer.writeheader()
#             writer.writerows(extracted_data)
        
#         print(f"Successfully extracted {len(extracted_data)} records to {output_file}")
        
#     except Exception as e:
#         print(f"Error writing CSV: {e}")

# Usage
if __name__ == "__main__":
    folder_path = input("Enter the folder path containing JSON files: ").strip()
    
    # Remove quotes if user pasted a path with quotes
    folder_path = folder_path.strip('"\'')
    
    # Choose which function to use:
    # Use this for one row per file (first result only)
    output_file ="/home/ethel/Documents/RESEARCH/SOCIAL_INTERACTIONS_2025/GERMAN_PUBLIC_POOLS/general_poi_json/reverse_geocoded_pool_pids_419.csv"
    extract_geocoding_data(folder_path, output_file)

    # Or use this for multiple rows per file (all results)
    # extract_all_geocoding_results(folder_path, 'geocoding_data.csv')