#!/usr/bin/env python3
# extract_place_ids_simple.py
import re
import os
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_place_ids_simple.py <html_directory>")
        print("Example: python extract_place_ids_simple.py google_maps_pages")
        sys.exit(1)
    
    html_dir = sys.argv[1]
    
    if not os.path.exists(html_dir):
        print(f"Error: Directory '{html_dir}' not found!")
        sys.exit(1)
    
    all_place_ids = set()
    
    print(f"Searching for Place IDs in {html_dir}/*.html")
    print("=" * 60)
    
    for filename in os.listdir(html_dir):
        if filename.endswith('.html'):
            filepath = os.path.join(html_dir, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Find all occurrences of your pattern
                pattern = r'placeid\\u003d([A-Za-z0-9_-]+)'
                matches = re.findall(pattern, content)
                
                if matches:
                    print(f"✓ {filename}: Found {len(matches)}")
                    for pid in matches:
                        all_place_ids.add(pid)
                        print(f"  → {pid}")
                else:
                    print(f"✗ {filename}: None found")
                    
            except Exception as e:
                print(f"! {filename}: Error - {e}")
    
    print("=" * 60)
    print(f"\n✅ Found {len(all_place_ids)} unique Place IDs")
    
    # Save results
    if all_place_ids:
        # Save to text file
        with open('place_ids.txt', 'w') as f:
            for pid in sorted(all_place_ids):
                f.write(f"{pid}\n")
        
        print(f"Saved to: place_ids.txt")
        
        # Also save with file associations
        with open('place_ids_detailed.txt', 'w') as f:
            for pid in sorted(all_place_ids):
                f.write(f"{pid}\n")
        
        # Show first 10 as sample
        print("\n📋 Sample of found Place IDs:")
        for i, pid in enumerate(sorted(all_place_ids)[:10]):
            print(f"  {i+1:2d}. {pid}")
        if len(all_place_ids) > 10:
            print(f"  ... and {len(all_place_ids) - 10} more")

if __name__ == "__main__":
    main()