import requests
import pandas as pd
import numpy as np
import time
import random
import json
import os
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import csv
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse
import urllib.parse
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('google_maps_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GoogleMapsScraper:
    """
    Specialized scraper for Google Maps pages with proxy support.
    """
    
    def __init__(self, proxy_file: Optional[str] = None, output_dir: str = 'google_maps_pages',
                 proxy_username: Optional[str] = None, proxy_password: Optional[str] = None):
        """
        Initialize the Google Maps scraper.
        
        Args:
            proxy_file: Path to proxy file (format: ip:port per line)
            output_dir: Directory to save downloaded HTML pages
            proxy_username: Username for proxy authentication
            proxy_password: Password for proxy authentication
        """
        self.output_dir = output_dir
        self.proxy_file = proxy_file
        
        # Authentication credentials (use provided or default)
        self.proxy_username = proxy_username or "iweber02"
        self.proxy_password = proxy_password or "qp9dQbDM"
        
        self.proxies = self._load_proxies() if proxy_file else []
        self.current_proxy_index = 0
        self.bad_proxies = set()  # Track proxies that failed
        
        # Google Maps specific headers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
        }
        
        # Create session for connection pooling
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Statistics
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'start_time': None,
            'end_time': None
        }
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def _load_proxies(self) -> List[Dict]:
        """
        Load proxies from file with authentication support.
        
        Returns:
            List of proxy dictionaries
        """
        proxies = []
        try:
            # URL encode the credentials
            username_encoded = urllib.parse.quote(self.proxy_username)
            password_encoded = urllib.parse.quote(self.proxy_password)
            
            with open(self.proxy_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Handle different formats
                    if '://' in line:
                        # Already has protocol (http://, https://, socks5://)
                        proxy_url = line
                    else:
                        # Simple ip:port format - add protocol and auth
                        if ':' in line:
                            proxy_url = f"http://{username_encoded}:{password_encoded}@{line}"
                        else:
                            logger.warning(f"Invalid proxy format on line {line_num}: {line}")
                            continue
                    
                    proxies.append({
                        'http': proxy_url,
                        'https': proxy_url,
                        'raw': line,  # Original line
                        'url': proxy_url,  # Full URL with auth
                        'index': line_num,
                        'success': 0,
                        'fail': 0
                    })
            
            if proxies:
                logger.info(f"Loaded {len(proxies)} proxies from {self.proxy_file}")
                
                # Log first few proxies (masked for security)
                sample_size = min(3, len(proxies))
                for i in range(sample_size):
                    proxy = proxies[i]
                    # Mask the password in logs
                    if '@' in proxy['url']:
                        parts = proxy['url'].split('@')
                        if len(parts) == 2:
                            masked_url = f"***@{parts[1]}"
                        else:
                            masked_url = "***@***"
                    else:
                        masked_url = proxy['url']
                    logger.info(f"  Proxy {i+1}: {masked_url}")
                
            else:
                logger.warning(f"No proxies found in {self.proxy_file}")
                
        except Exception as e:
            logger.error(f"Error loading proxies: {e}")
            
        return proxies
    
    def get_next_proxy(self) -> Optional[Dict]:
        """
        Get next proxy in rotation, skipping bad ones.
        
        Returns:
            Proxy dictionary or None if no proxies available
        """
        if not self.proxies:
            return None
        
        # If all proxies marked bad, reset and try again
        if len(self.bad_proxies) >= len(self.proxies):
            logger.warning("All proxies marked bad, resetting...")
            self.bad_proxies.clear()
        
        # Try to get a good proxy
        attempts = 0
        while attempts < len(self.proxies):
            proxy = self.proxies[self.current_proxy_index]
            self.current_proxy_index = (self.current_proxy_index + 1) % len(self.proxies)
            
            if proxy['url'] not in self.bad_proxies:
                return proxy
            
            attempts += 1
        
        logger.error("No working proxies available!")
        return None
    
    def mark_proxy_bad(self, proxy_url: str):
        """Mark a proxy as bad."""
        self.bad_proxies.add(proxy_url)
        
    def mark_proxy_good(self, proxy_url: str):
        """Mark a proxy as good (remove from bad list)."""
        if proxy_url in self.bad_proxies:
            self.bad_proxies.remove(proxy_url)
    
    def _extract_place_id(self, url: str) -> str:
        """
        Extract a unique identifier from Google Maps URL.
        
        Args:
            url: Google Maps URL
            
        Returns:
            Extracted place ID or generated identifier
        """
        try:
            # Try to extract from URL parameters
            parsed = urlparse(url)
            if '/place/' in url:
                # Extract place name from URL path
                path_parts = parsed.path.split('/')
                if len(path_parts) > 2 and path_parts[1] == 'place':
                    return path_parts[2].replace('+', '_')
            
            # Try to extract from data parameter
            if '/data=' in url:
                import re
                match = re.search(r'!1s([^!:]+)', url)
                if match:
                    return match.group(1).replace(':', '_')
            
            # Fallback: use domain and timestamp
            domain = parsed.netloc.replace('www.', '').replace('.google.com', '').replace('.', '_')
            return f"{domain}_{int(time.time())}"
            
        except:
            return f"place_{int(time.time())}"
    
    def download_page(self, url: str, pool_id: str, metadata: Dict, 
                     max_retries: int = 3) -> Dict:
        """
        Download a single Google Maps page.
        
        Args:
            url: Google Maps URL to download
            pool_id: Unique identifier for the pool
            metadata: Dictionary with pool metadata
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dictionary with download results
        """
        result = {
            'pool_id': pool_id,
            'url': url,
            'success': False,
            'filename': None,
            'proxy_used': None,
            'status_code': None,
            'error': None,
            'retries': 0,
            'download_time': None,
            'timestamp': datetime.now().isoformat()
        }
        
        start_time = time.time()
        
        for attempt in range(max_retries):
            try:
                # Get proxy for this attempt
                proxies_config = None
                proxy_dict = None
                if self.proxies:
                    proxy_dict = self.get_next_proxy()
                    if proxy_dict:
                        proxies_config = {
                            'http': proxy_dict['url'],
                            'https': proxy_dict['url']
                        }
                        # Mask proxy in logs for security
                        proxy_display = proxy_dict['raw']
                        result['proxy_used'] = proxy_display
                
                # Add random delay to avoid rate limiting
                if attempt > 0:
                    delay = random.uniform(2 ** attempt, 2 ** (attempt + 1))
                    logger.info(f"Retry {attempt + 1}/{max_retries} for pool {pool_id}, waiting {delay:.1f}s")
                    time.sleep(delay)
                else:
                    # Random delay between 1-3 seconds for first attempt
                    time.sleep(random.uniform(1, 3))
                
                # Make the request
                response = self.session.get(
                    url,
                    proxies=proxies_config,
                    timeout=30,
                    allow_redirects=True
                )
                
                result['status_code'] = response.status_code
                result['retries'] = attempt
                
                if response.status_code == 200:
                    # Create filename
                    safe_pool_id = str(pool_id).replace('/', '_').replace('\\', '_')
                    filename = f"pool_{safe_pool_id}_{self._extract_place_id(url)}.html"
                    filepath = os.path.join(self.output_dir, filename)
                    
                    # Save HTML
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(response.text)
                    
                    # Save metadata as JSON
                    metadata_filename = filepath.replace('.html', '_metadata.json')
                    metadata['download_timestamp'] = datetime.now().isoformat()
                    metadata['original_url'] = url
                    metadata['html_filename'] = filename
                    
                    with open(metadata_filename, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)
                    
                    result.update({
                        'success': True,
                        'filename': filename,
                        'file_size': len(response.text),
                        'download_time': time.time() - start_time
                    })
                    
                    # Mark proxy as good
                    if proxy_dict:
                        self.mark_proxy_good(proxy_dict['url'])
                        proxy_dict['success'] += 1
                    
                    logger.info(f"✓ Downloaded pool {pool_id}: {metadata.get('pool_name', 'Unknown')}")
                    return result
                    
                else:
                    result['error'] = f"HTTP {response.status_code}"
                    logger.warning(f"HTTP {response.status_code} for pool {pool_id}")
                    
                    # Mark proxy as bad for certain status codes
                    if proxy_dict and response.status_code in [403, 407, 429, 503]:
                        self.mark_proxy_bad(proxy_dict['url'])
                        proxy_dict['fail'] += 1
                    
            except requests.exceptions.ProxyError as e:
                result['error'] = str(e)
                logger.warning(f"Proxy error for pool {pool_id}: {e}")
                
                # Mark proxy as bad
                if proxy_dict:
                    self.mark_proxy_bad(proxy_dict['url'])
                    proxy_dict['fail'] += 1
                    
            except requests.exceptions.RequestException as e:
                result['error'] = str(e)
                logger.warning(f"Attempt {attempt + 1} failed for pool {pool_id}: {e}")
                
            except Exception as e:
                result['error'] = str(e)
                logger.error(f"Unexpected error for pool {pool_id}: {e}")
                break
        
        result['download_time'] = time.time() - start_time
        logger.error(f"✗ Failed to download pool {pool_id} after {max_retries} attempts")
        return result
    
    def process_dataset(self, dataset_path: str, max_workers: int = 5,
                       batch_size: Optional[int] = None) -> pd.DataFrame:
        """
        Process all URLs from the dataset.
        """
        logger.info(f"Starting Google Maps scraper")
        logger.info(f"Dataset: {dataset_path}")
        logger.info(f"Proxies: {len(self.proxies) if self.proxies else 'No proxies'}")
        logger.info(f"Output dir: {self.output_dir}")
        
        self.stats['start_time'] = datetime.now()
        
        # Load dataset
        try:
            # Read with pool_id as string to avoid type issues
            df = pd.read_csv(dataset_path, dtype={'pool_id': str})
            logger.info(f"Loaded dataset with {len(df)} rows")
            
            # Filter for valid URLs
            if 'place_page_url' not in df.columns:
                logger.error("Dataset must contain 'place_page_url' column")
                return pd.DataFrame()
            
            # Apply batch limit if specified
            if batch_size and batch_size < len(df):
                df = df.head(batch_size)
                logger.info(f"Processing first {batch_size} rows (batch mode)")
            
            # Ensure pool_id is string type
            if 'pool_id' in df.columns:
                df['pool_id'] = df['pool_id'].astype(str)
            
            self.stats['total'] = len(df)
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return pd.DataFrame()
        
        # Prepare results list
        all_results = []
        
        # Process URLs
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            future_to_row = {}
            for idx, row in df.iterrows():
                url = str(row['place_page_url']).strip()
                
                # Skip empty or invalid URLs
                if not url or not url.startswith('http'):
                    logger.warning(f"Skipping row {idx}: invalid URL")
                    continue
                
                # Get pool_id as string
                pool_id = str(row.get('pool_id', idx))
                
                # Create metadata from row
                metadata = row.to_dict()
                
                # Submit download task
                future = executor.submit(
                    self.download_page, 
                    url, 
                    pool_id, 
                    metadata
                )
                future_to_row[future] = (pool_id, row.get('pool_name', 'Unknown'))
            
            # Process completed tasks
            for future in as_completed(future_to_row):
                pool_id, pool_name = future_to_row[future]
                try:
                    result = future.result(timeout=60)
                    all_results.append(result)
                    
                    if result['success']:
                        self.stats['success'] += 1
                    else:
                        self.stats['failed'] += 1
                        
                except Exception as e:
                    logger.error(f"Error processing pool {pool_id} ({pool_name}): {e}")
                    self.stats['failed'] += 1
        
        # Create results DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Ensure pool_id is string in results
        if 'pool_id' in results_df.columns:
            results_df['pool_id'] = results_df['pool_id'].astype(str)
        
        # Save detailed results
        self._save_results(results_df, df)
        
        # Print summary
        self._print_summary()
        
        return results_df
    def _save_results(self, results_df: pd.DataFrame, original_df: pd.DataFrame):
        """
        Save download results to files.
        
        Args:
            results_df: DataFrame with download results
            original_df: Original dataset DataFrame
        """
        if not results_df.empty:
            # Convert pool_id to string in both DataFrames to ensure consistent types
            original_df = original_df.copy()
            results_df = results_df.copy()
            
            # Ensure pool_id is string type for both
            original_df['pool_id'] = original_df['pool_id'].astype(str)
            results_df['pool_id'] = results_df['pool_id'].astype(str)
            
            # Merge with original data
            merged_df = original_df.merge(
                results_df,
                left_on='pool_id',
                right_on='pool_id',
                how='left',
                suffixes=('_original', '_download')
            )
            
            # Save to CSV
            results_csv = os.path.join(self.output_dir, 'download_results.csv')
            merged_df.to_csv(results_csv, index=False, encoding='utf-8')
            logger.info(f"Results saved to: {results_csv}")
            
            # Save summary statistics
            summary = {
                'total_urls': int(self.stats['total']),
                'successful': int(self.stats['success']),
                'failed': int(self.stats['failed']),
                'success_rate': float(self.stats['success'] / self.stats['total'] * 100) if self.stats['total'] > 0 else 0,
                'start_time': self.stats['start_time'].isoformat() if self.stats['start_time'] else None,
                'end_time': datetime.now().isoformat(),
                'output_directory': os.path.abspath(self.output_dir),
                'proxies_used': len(self.proxies) if self.proxies else 0,
                'working_proxies': len(self.proxies) - len(self.bad_proxies) if self.proxies else 0,
                'bad_proxies': len(self.bad_proxies)
            }
            
            summary_json = os.path.join(self.output_dir, 'summary.json')
            with open(summary_json, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Summary saved to: {summary_json}")
            
            # Save failed URLs for retry
            failed_df = results_df[~results_df['success']]
            if not failed_df.empty:
                failed_csv = os.path.join(self.output_dir, 'failed_urls.csv')
                failed_df[['pool_id', 'url', 'error', 'status_code']].to_csv(failed_csv, index=False)
                logger.info(f"Failed URLs saved to: {failed_csv}")
            
            # Also save results separately without merge
            simple_results_csv = os.path.join(self.output_dir, 'simple_results.csv')
            results_df.to_csv(simple_results_csv, index=False, encoding='utf-8')
            logger.info(f"Simple results saved to: {simple_results_csv}")

    def _print_summary(self):
        """Print summary of the scraping session."""
        duration = datetime.now() - self.stats['start_time'] if self.stats['start_time'] else None
        duration_str = f"{duration.total_seconds():.1f}s" if duration else "N/A"
        
        print("\n" + "="*70)
        print("GOOGLE MAPS SCRAPING SUMMARY")
        print("="*70)
        print(f"Total URLs processed: {self.stats['total']}")
        print(f"Successfully downloaded: {self.stats['success']}")
        print(f"Failed: {self.stats['failed']}")
        
        if self.stats['total'] > 0:
            success_rate = (self.stats['success'] / self.stats['total']) * 100
            print(f"Success rate: {success_rate:.1f}%")
        
        print(f"Duration: {duration_str}")
        print(f"Output directory: {os.path.abspath(self.output_dir)}")
        
        if self.proxies:
            print(f"Proxies used: {len(self.proxies)}")
            print(f"Working proxies: {len(self.proxies) - len(self.bad_proxies)}")
            print(f"Bad proxies: {len(self.bad_proxies)}")
        
        print("="*70)
        
        # Show sample of downloaded files
        if os.path.exists(self.output_dir):
            html_files = [f for f in os.listdir(self.output_dir) if f.endswith('.html')]
            if html_files:
                print(f"\nSample of downloaded files ({len(html_files)} total):")
                for f in html_files[:5]:
                    filepath = os.path.join(self.output_dir, f)
                    size_kb = os.path.getsize(filepath) / 1024
                    print(f"  • {f} ({size_kb:.1f} KB)")
                if len(html_files) > 5:
                    print(f"  ... and {len(html_files) - 5} more")

def validate_dataset(dataset_path: str) -> bool:
    """
    Validate the dataset file.
    
    Args:
        dataset_path: Path to dataset CSV file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        df = pd.read_csv(dataset_path)
        
        # Check required columns
        required_columns = ['place_page_url', 'pool_id']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check for valid URLs
        valid_urls = df['place_page_url'].apply(
            lambda x: isinstance(x, str) and x.startswith('http')
        ).sum()
        
        logger.info(f"Found {valid_urls} valid URLs out of {len(df)} rows")
        
        # Show sample
        print("\nDataset preview:")
        print(df[['pool_id', 'pool_name', 'place_page_url']].head().to_string())
        
        return True
        
    except Exception as e:
        logger.error(f"Dataset validation failed: {e}")
        return False

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description='Google Maps Scraper - Download HTML pages from Google Maps URLs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with dataset only
  python google_maps_scraper.py --dataset pools.csv
  
  # With proxies (using default authentication)
  python google_maps_scraper.py --dataset pools.csv --proxies proxies.txt
  
  # Custom output directory
  python google_maps_scraper.py --dataset pools.csv --output my_pages
  
  # Test with first 10 entries
  python google_maps_scraper.py --dataset pools.csv --batch-size 10
  
  # More workers for faster scraping
  python google_maps_scraper.py --dataset pools.csv --workers 10
  
  # Custom proxy authentication
  python google_maps_scraper.py --dataset pools.csv --proxies proxies.txt --proxy-user myuser --proxy-pass mypass
        """
    )
    
    parser.add_argument('--dataset', '-d', required=True,
                       help='Path to CSV dataset file (required)')
    parser.add_argument('--proxies', '-p',
                       help='Path to proxy file (one proxy per line, format: ip:port)')
    parser.add_argument('--output', '-o', default='google_maps_pages',
                       help='Output directory (default: google_maps_pages)')
    parser.add_argument('--workers', '-w', type=int, default=5,
                       help='Number of concurrent workers (default: 5)')
    parser.add_argument('--batch-size', '-b', type=int,
                       help='Process only first N rows (for testing)')
    parser.add_argument('--proxy-user', type=str, default='iweber02',
                       help='Proxy username (default: iweber02)')
    parser.add_argument('--proxy-pass', type=str, default='qp9dQbDM',
                       help='Proxy password (default: qp9dQbDM)')
    parser.add_argument('--validate', action='store_true',
                       help='Validate dataset only (don\'t scrape)')
    parser.add_argument('--no-proxies', action='store_true',
                       help='Run without proxies even if proxy file exists')
    
    args = parser.parse_args()
    
    # Validate dataset
    if not validate_dataset(args.dataset):
        print("Dataset validation failed. Please check your CSV file.")
        return
    
    if args.validate:
        print("Dataset validation passed.")
        return
    
    # Initialize scraper
    proxy_file = None if args.no_proxies else args.proxies
    
    scraper = GoogleMapsScraper(
        proxy_file=proxy_file,
        output_dir=args.output,
        proxy_username=args.proxy_user,
        proxy_password=args.proxy_pass
    )
    
    # Process dataset
    print(f"\n{'='*70}")
    print("STARTING GOOGLE MAPS SCRAPER")
    print(f"{'='*70}")
    
    results = scraper.process_dataset(
        dataset_path=args.dataset,
        max_workers=args.workers,
        batch_size=args.batch_size
    )
    
    if not results.empty:
        print(f"\n✅ Scraping completed!")
        print(f"   Results saved in: {os.path.abspath(args.output)}")
        print(f"   Check 'download_results.csv' for detailed results")
    else:
        print(f"\n❌ Scraping failed or no URLs processed")

if __name__ == "__main__":
    main()