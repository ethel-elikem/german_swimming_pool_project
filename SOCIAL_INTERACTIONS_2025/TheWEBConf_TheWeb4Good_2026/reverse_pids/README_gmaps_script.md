# Google Maps URL Extractor for Swimming Pools

This script automates the process of searching swimming pools on Google Maps and extracting their unique identifiers.

## Features

- Searches each pool on Google Maps using name, address, or both
- Automatically clicks the first result if landing on a search page
- Extracts the full Google Maps place URL
- Extracts unique identifier from URL using regex patterns
- Saves intermediate results every 10 rows
- Detailed logging of the search process

## Requirements

- Python 3.7+
- Chrome browser installed
- ChromeDriver (will be managed automatically by selenium)

## Installation

1. Install the required packages:
```bash
pip install -r requirements_gmaps.txt
```

Or install manually:
```bash
pip install pandas selenium webdriver-manager
```

## Usage

### Basic usage (headless mode):
```bash
python get_gmaps_urls.py
```

### With custom input/output files:
```bash
python get_gmaps_urls.py --input 843_pools.csv --output pools_with_urls.csv
```

### Run with visible browser (for debugging):
```bash
python get_gmaps_urls.py --visible
```

### All options:
```bash
python get_gmaps_urls.py --input INPUT_FILE --output OUTPUT_FILE [--visible]
```

## Parameters

- `--input`: Input CSV file path (default: `843_pools.csv`)
- `--output`: Output CSV file path (default: `843_pools_with_urls.csv`)
- `--headless`: Run in headless mode (default: True)
- `--visible`: Run with visible browser for debugging

## Input CSV Format

The script expects a CSV file with at least these columns:
- `pool_id`: Unique identifier for the pool
- `pool_name`: Name of the swimming pool
- `address_admin`: Full address of the pool

## Output CSV Format

The output CSV contains all original columns plus:
- `url`: The full Google Maps place URL
- `unique_identifier`: The extracted identifier (format: `0xHEX:0xHEX`)

## How It Works

1. **Search Strategy**: For each pool, the script tries:
   - Name + Address
   - Name only
   - Address only

2. **URL Validation**: Only URLs starting with `https://www.google.com/maps/place` are considered successful

3. **Identifier Extraction**: Uses two regex patterns:
   - Strict: `(?!0x0:0x)0x[0-9a-f]+:0x[0-9a-f]+` (excludes 0x0:0x)
   - Relaxed: `0x[0-9a-f]+:0x[0-9a-f]+` (if strict fails)

4. **Error Handling**: 
   - Saves progress every 10 rows
   - Continues processing even if individual searches fail
   - Logs all activities for debugging

## Troubleshooting

### ChromeDriver Issues
If you encounter ChromeDriver issues, you can install it manually:
```bash
# Using webdriver-manager (recommended)
pip install webdriver-manager

# The script will automatically download the correct driver
```

### Rate Limiting
The script includes a 2-second delay between searches. If you encounter issues:
- Increase the delay in the code (line with `time.sleep(2)`)
- Run in smaller batches
- Use `--visible` to see what's happening

### No Results Found
If many searches fail:
1. Check your internet connection
2. Verify the input data format
3. Run with `--visible` to see browser behavior
4. Check the log output for specific errors

## Example Output

```
2026-02-06 10:00:00 - INFO - Reading input CSV: 843_pools.csv
2026-02-06 10:00:05 - INFO - Processing pool 1/843: ID=37
2026-02-06 10:00:05 - INFO - Name: Freibad Hangeweiher
2026-02-06 10:00:05 - INFO - Address: Am Hangeweiher 32, 52074 Aachen, Germany
2026-02-06 10:00:08 - INFO - Success! Found place URL with ID: 0x47c099d5e2b5c123:0x8f4a5b6c7d8e9f0a
...
```

## Notes

- The script is designed for German swimming pools but can work with any location
- Processing 843 pools will take approximately 30-60 minutes
- Results are saved incrementally to prevent data loss
