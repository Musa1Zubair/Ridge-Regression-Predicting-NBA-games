import os
import asyncio
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
import time
import traceback

SEASONS = list(range(2025, 2026))
DATA_DIR = "data"
STANDINGS_DIR = os.path.join(DATA_DIR, "standings")
SCORES_DIR = os.path.join(DATA_DIR, "scores")

async def get_html(url, selector, sleep=5, retries=3):
    html = None
    for i in range(1, retries+1):
        time.sleep(sleep * i)
        try:
            async with async_playwright() as p:
                browser = await p.firefox.launch()
                page = await browser.new_page()
                await page.goto(url)
                print(await page.title())
                html = await page.inner_html(selector)
        except PlaywrightTimeout:
            print(f"Timeout error on {url}")
            continue
        else:
            break
    return html


async def scrape_season(season):
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games.html"
    html = await get_html(url, "#content .filter")
    
    soup = BeautifulSoup(html)
    links = soup.find_all("a")
    standings_pages = [f"https://www.basketball-reference.com{l['href']}" for l in links]
    
    for url in standings_pages:
        save_path = os.path.join(STANDINGS_DIR, url.split("/")[-1])
        if os.path.exists(save_path):
            continue
        
        html = await get_html(url, "#all_schedule")
        with open(save_path, "w+", encoding="utf-8") as f:
            f.write(html)


async def scrape_game(standings_file):
    with open(standings_file, 'r', encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html)
    links = soup.find_all("a")
    hrefs = [l.get('href') for l in links]
    box_scores = [f"https://www.basketball-reference.com{l}" for l in hrefs if l and "boxscore" in l and '.html' in l]

    for url in box_scores:
        save_path = os.path.join(SCORES_DIR, url.split("/")[-1])
        if os.path.exists(save_path):
            continue

        html = await get_html(url, "#content")
        if not html:
            continue
        with open(save_path, "w+", encoding="utf-8") as f:
            f.write(html)


async def main_loop():
    os.makedirs(STANDINGS_DIR, exist_ok=True)
    os.makedirs(SCORES_DIR, exist_ok=True)
    

    for season in SEASONS:
        await scrape_season(season)
    
    
    standings_files = os.listdir(STANDINGS_DIR)
    for f in standings_files:
        filepath = os.path.join(STANDINGS_DIR, f)
        try:
            await scrape_game(filepath)
        except Exception as e:
            print(f"Error processing {filepath}: {str(e)}")
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main_loop())
