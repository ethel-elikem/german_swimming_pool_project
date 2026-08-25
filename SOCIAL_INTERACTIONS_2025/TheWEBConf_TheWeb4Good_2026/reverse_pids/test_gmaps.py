import time
import os
import pandas as pd
from urllib.parse import urlparse, parse_qs, unquote

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, TimeoutException
# 1. Add explicit waits for content
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ---------- Config ----------
INPUT_CSV = "/home/ethel/Documents/RESEARCH/SOCIAL_INTERACTIONS_2025/TheWEBConf_TheWeb4Good_2026/reverse_pids/843_pools_with_search_links.csv"
OUTPUT_CSV = "/home/ethel/Documents/RESEARCH/SOCIAL_INTERACTIONS_2025/TheWEBConf_TheWeb4Good_2026/reverse_pids/843_pools_html.csv"
COLUMN_NAME = "google_maps_search_link"

HTML_OUTPUT_DIR = "final_html"   # folder where HTML files will be saved

CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/130.0.0.0 Safari/537.36"
)

MAX_ROWS = 5 # set to a large number to process all rows
LOAD_WAIT = 4
AFTER_CLICK_WAIT = 3


# Make sure folder exists
os.makedirs(HTML_OUTPUT_DIR, exist_ok=True) 


# ---------- Helper: fallback parsing of consent URLs ----------
def extract_continue_from_consent(url: str) -> str:
    parsed = urlparse(url)
    if "consent.google.com" not in parsed.netloc:
        return url

    qs = parse_qs(parsed.query)
    if "continue" not in qs:
        return url

    cont = qs["continue"][0]
    return unquote(unquote(cont))


# ---------- Selenium setup ----------
def create_driver():
    options = webdriver.ChromeOptions()
    options.add_argument(f"user-agent={CHROME_UA}")
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=options)

    driver.get("https://www.google.com")
    time.sleep(2)

    try:
        driver.add_cookie({
            "name": "CONSENT",
            "value": "YES+cb.20210328-17-p0.en+FX+123",
            "domain": ".google.com",
            "path": "/"
        })
    except Exception:
        pass

    return driver


# ---------- Consent page handler ----------
def handle_consent_if_present(driver):
    if "consent.google.com" not in driver.current_url:
        return

    xpaths = [
        "//button[contains(., 'Ich stimme zu')]",
        "//button[contains(., 'Alle akzeptieren')]",
        "//button[contains(., 'Ich akzeptiere')]",
        "//button[contains(., 'Accept all')]",
        "//button[contains(., 'I agree')]",
        "//button[@aria-label='Accept all']",
        "//button[@aria-label='I agree']",
    ]

    for xp in xpaths:
        try:
            driver.find_element(By.XPATH, xp).click()
            time.sleep(AFTER_CLICK_WAIT)
            return
        except Exception:
            continue


# ---------- Resolve URL with Selenium ----------
def resolve_final_url_selenium(driver, url: str) -> str:
    try:
        driver.get(url)
        time.sleep(LOAD_WAIT)

        if "consent.google.com" in driver.current_url:
            handle_consent_if_present(driver)
            time.sleep(LOAD_WAIT)

        final_url = driver.current_url

        if "consent.google.com" in final_url:
            final_url = extract_continue_from_consent(final_url)

        return final_url

    except Exception as e:
        return f"ERROR: {e}"


# ---------- Main script ----------
def main():
    df = pd.read_csv(INPUT_CSV, sep=None, engine="python")

    if COLUMN_NAME not in df.columns:
        raise ValueError(f"Column '{COLUMN_NAME}' not found. Columns: {df.columns.tolist()}")

    driver = create_driver()

    try:
        n = len(df)
        rows_to_process = min(MAX_ROWS, n)
        final_urls = []
        html_files = []

        for i in range(rows_to_process):
            raw_url = df.loc[i, COLUMN_NAME]
            print(f"{i+1}/{rows_to_process} – Resolving: {raw_url}")

            if pd.isna(raw_url) or str(raw_url).strip() == "":
                final_urls.append(None)
                html_files.append(None)
                continue

            final_url = resolve_final_url_selenium(driver, str(raw_url))
            print("   → Final URL:", final_url)

            final_urls.append(final_url)
            html_filename = None

            # save HTML only if final URL is valid and does NOT include "search"
            if (
                isinstance(final_url, str) and
                "ERROR" not in final_url and
                "search" not in final_url
            ):
                html = driver.page_source
                html_filename = f"row_{i}_final.html"
                filepath = os.path.join(HTML_OUTPUT_DIR, html_filename)

                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(html)

                print(f"   → Saved HTML: {filepath}")

            html_files.append(html_filename)

        df["final_url"] = [final_urls[i] if i < rows_to_process else None for i in range(n)]
        df["html_file"] = [html_files[i] if i < rows_to_process else None for i in range(n)]

        df.to_csv(OUTPUT_CSV, index=False)
        print("Done. Saved to:", OUTPUT_CSV)

    finally:
        driver.quit()


if __name__ == "__main__":
    main()