"""
Test script to verify Google Maps consent handling works properly.
Runs with visible browser so you can see what's happening.
"""

import pandas as pd
import sys
sys.path.insert(0, '.')

from get_gmaps_urls_2026 import GoogleMapsSearcher
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_consent_and_search():
    """Test the consent handling and a single search."""
    
    # Read just the first row for testing
    df = pd.read_csv('843_pools.csv')
    test_row = df.iloc[0]
    
    print("\n" + "="*60)
    print("TESTING WITH VISIBLE BROWSER")
    print("="*60)
    print(f"Pool Name: {test_row['pool_name']}")
    print(f"Address: {test_row['address_admin']}")
    print("="*60 + "\n")
    
    # Initialize searcher with visible browser
    searcher = GoogleMapsSearcher(headless=False)
    
    try:
        # Search for the place
        url, unique_id = searcher.search_place(
            test_row['pool_name'],
            test_row['address_admin']
        )
        
        print("\n" + "="*60)
        print("RESULTS:")
        print("="*60)
        print(f"URL: {url}")
        print(f"Unique ID: {unique_id}")
        print("="*60 + "\n")
        
        if url:
            print("✓ SUCCESS! Consent handling worked and place was found!")
        else:
            print("✗ FAILED: Could not find the place")
            
        # Keep browser open for a few seconds so you can see the result
        input("\nPress Enter to close the browser...")
        
    finally:
        searcher.close()

if __name__ == "__main__":
    test_consent_and_search()
