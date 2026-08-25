"""
Script to search swimming pools on Google Maps and extract place identifiers.
Searches each pool by name and/or address, retrieves the Google Maps URL,
and extracts the unique identifier from the URL.
"""

import pandas as pd
import re
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from urllib.parse import quote_plus
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GoogleMapsSearcher:
    """Class to handle Google Maps searches and URL extraction."""
    
    def __init__(self, headless=True):
        """Initialize the Chrome driver."""
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        # Add user agent to appear more like a regular browser
        options.add_argument('--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        # Accept language
        options.add_argument('--lang=en-US')
        
        self.driver = webdriver.Chrome(options=options)
        self.driver.implicitly_wait(10)
        
        # Patterns for extracting unique identifier
        self.strict_pattern = r'(?!0x0:0x)0x[0-9a-f]+:0x[0-9a-f]+'
        self.relaxed_pattern = r'0x[0-9a-f]+:0x[0-9a-f]+'
        
        # Handle consent on first visit
        self.consent_handled = False
        self._handle_initial_consent()
    
    def _handle_initial_consent(self):
        """Handle Google consent page on first visit."""
        try:
            logger.info("Navigating to Google Maps to handle consent...")
            self.driver.get("https://www.google.com/maps")
            time.sleep(3)
            
            # Check if we're on a consent page
            if "consent.google.com" in self.driver.current_url:
                logger.info("Consent page detected, attempting to accept...")
                
                # Try different consent button selectors
                consent_selectors = [
                    "button[aria-label*='Accept']",
                    "button[aria-label*='Agree']",
                    "button[aria-label*='accept']",
                    "button[aria-label*='agree']",
                    "form[action*='consent'] button",
                    "button[type='submit']",
                    ".VfPpkd-LgbsSe",  # Google Material Design button class
                ]
                
                for selector in consent_selectors:
                    try:
                        buttons = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        for button in buttons:
                            if button.is_displayed():
                                logger.info(f"Found consent button with selector: {selector}")
                                button.click()
                                time.sleep(3)
                                logger.info("Consent accepted!")
                                self.consent_handled = True
                                return
                    except Exception:
                        continue
                
                # If no button found, try to skip by navigating directly
                logger.warning("Could not find consent button, trying direct navigation...")
                self.driver.get("https://www.google.com/maps")
                time.sleep(3)
                
            logger.info("Consent handling complete.")
            self.consent_handled = True
            
        except Exception as e:
            logger.error(f"Error handling consent: {e}")
            self.consent_handled = True  # Continue anyway
    
    def _check_and_handle_consent(self):
        """Check if we're on a consent page and handle it."""
        if "consent.google.com" in self.driver.current_url:
            logger.info("Consent page detected during search...")
            
            try:
                # Try to click accept/agree button
                consent_selectors = [
                    "button[aria-label*='Accept']",
                    "button[aria-label*='Agree']",
                    "form[action*='consent'] button",
                    ".VfPpkd-LgbsSe",
                ]
                
                for selector in consent_selectors:
                    try:
                        button = WebDriverWait(self.driver, 2).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                        )
                        button.click()
                        time.sleep(2)
                        logger.info("Consent accepted during search")
                        return True
                    except Exception:
                        continue
                        
            except Exception as e:
                logger.warning(f"Could not handle consent: {e}")
            
            return False
        return True
    
    def search_place(self, pool_name, address):
        """
        Search for a place on Google Maps using name and/or address.
        
        Args:
            pool_name: Name of the swimming pool
            address: Address of the swimming pool
            
        Returns:
            tuple: (url, unique_identifier) or (None, None) if not found
        """
        search_queries = []
        
        # Try different search combinations
        if pool_name and address:
            search_queries.append(f"{pool_name}, {address}")
        if pool_name:
            search_queries.append(pool_name)
        if address:
            search_queries.append(address)
        
        for query in search_queries:
            logger.info(f"Searching: {query}")
            
            try:
                # Navigate to Google Maps
                encoded_query = quote_plus(query)
                search_url = f"https://www.google.com/maps/search/{encoded_query}"
                self.driver.get(search_url)
                
                # Wait for page to load
                time.sleep(3)
                
                # Check and handle consent if needed
                self._check_and_handle_consent()
                
                # Wait a bit more after consent
                time.sleep(2)
                
                # Get the current URL after redirects
                current_url = self.driver.current_url
                logger.info(f"Current URL: {current_url}")
                
                # Check if we got a place page (URL starts with the expected pattern)
                if current_url.startswith("https://www.google.com/maps/place"):
                    # Extract unique identifier
                    unique_id = self.extract_identifier(current_url)
                    logger.info(f"Success! Found place URL with ID: {unique_id}")
                    return current_url, unique_id
                
                # If we're on a search results page, try clicking the first result
                if "/search/" in current_url:
                    try:
                        # Wait for search results to load
                        WebDriverWait(self.driver, 5).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='/maps/place/']"))
                        )
                        
                        # Click the first result
                        first_result = self.driver.find_element(By.CSS_SELECTOR, "a[href*='/maps/place/']")
                        first_result.click()
                        
                        # Wait for navigation
                        time.sleep(3)
                        
                        current_url = self.driver.current_url
                        if current_url.startswith("https://www.google.com/maps/place"):
                            unique_id = self.extract_identifier(current_url)
                            logger.info(f"Success after clicking first result! ID: {unique_id}")
                            return current_url, unique_id
                    except (TimeoutException, NoSuchElementException) as e:
                        logger.warning(f"No clickable results found: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error searching for '{query}': {e}")
                continue
        
        logger.warning(f"No valid place URL found for {pool_name}")
        return None, None
    
    def extract_identifier(self, url):
        """
        Extract the unique identifier from Google Maps URL.
        Tries strict pattern first, then relaxed pattern.
        
        Args:
            url: Google Maps URL
            
        Returns:
            str: Unique identifier or None if not found
        """
        if not url:
            return None
        
        # Try strict pattern first (excludes 0x0:0x)
        match = re.search(self.strict_pattern, url)
        if match:
            return match.group(0)
        
        # Try relaxed pattern
        match = re.search(self.relaxed_pattern, url)
        if match:
            identifier = match.group(0)
            # Check if it's not 0x0:0x
            if identifier != "0x0:0x0":
                return identifier
        
        return None
    
    def close(self):
        """Close the browser."""
        if self.driver:
            self.driver.quit()


def process_pools_csv(input_csv, output_csv, headless=True):
    """
    Process the swimming pools CSV file.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file
        headless: Whether to run browser in headless mode
    """
    # Read the input CSV
    logger.info(f"Reading input CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Add new columns
    df['url'] = None
    df['unique_identifier'] = None
    
    # Initialize the searcher
    searcher = GoogleMapsSearcher(headless=headless)
    
    try:
        # Process each row
        for idx, row in df.iterrows():
            pool_id = row.get('pool_id', idx)
            pool_name = row.get('pool_name', '')
            address = row.get('address_admin', '')
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing pool {idx+1}/{len(df)}: ID={pool_id}")
            logger.info(f"Name: {pool_name}")
            logger.info(f"Address: {address}")
            
            # Search for the place
            url, unique_id = searcher.search_place(pool_name, address)
            
            # Update the dataframe
            df.at[idx, 'url'] = url
            df.at[idx, 'unique_identifier'] = unique_id
            
            # Save intermediate results every 10 rows
            if (idx + 1) % 10 == 0:
                logger.info(f"Saving intermediate results...")
                df.to_csv(output_csv, index=False)
            
            # Add a small delay to avoid rate limiting
            time.sleep(2)
    
    finally:
        # Close the browser
        searcher.close()
    
    # Save final results
    logger.info(f"\n{'='*60}")
    logger.info(f"Saving final results to: {output_csv}")
    df.to_csv(output_csv, index=False)
    
    # Print summary
    successful = df['url'].notna().sum()
    with_identifier = df['unique_identifier'].notna().sum()
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing complete!")
    logger.info(f"Total pools: {len(df)}")
    logger.info(f"Successful URL retrievals: {successful}")
    logger.info(f"Unique identifiers extracted: {with_identifier}")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Search swimming pools on Google Maps')
    parser.add_argument(
        '--input',
        type=str,
        default='843_pools.csv',
        help='Input CSV file path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='843_pools_with_urls.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        default=True,
        help='Run browser in headless mode'
    )
    parser.add_argument(
        '--visible',
        action='store_true',
        help='Run browser in visible mode (not headless)'
    )
    
    args = parser.parse_args()
    
    # Determine headless mode
    headless = args.headless and not args.visible
    
    # Process the CSV
    process_pools_csv(args.input, args.output, headless=headless)
