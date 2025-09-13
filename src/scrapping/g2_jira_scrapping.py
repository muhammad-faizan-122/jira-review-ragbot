import asyncio
import json
import os
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy
from crawl4ai.cache_context import CacheMode


def save_data(data, filename="reviews.json"):
    """Saves the scraped data to a JSON file, creating the directory if needed."""
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Data saved to {filename}")


async def scrape_g2_reviews():
    url = "https://www.g2.com/products/jira/reviews"
    session_id = "g2_full_scrape_session"
    all_reviews = []
    page_number = 1

    # --- JavaScript Snippets (Unchanged) ---

    js_next_page_and_save_state = """
    const firstReview = document.querySelector('div[itemprop="name"] div');
    if (firstReview) {
        window.lastReviewTitle = firstReview.textContent.trim();
    }
    const button = document.querySelector('a.pagination__named-link');
    if (button) { button.click(); }
    """

    wait_for_new_reviews = """
    () => {
        const firstReview = document.querySelector('div[itemprop="name"] div');
        if (!firstReview) return false;
        const firstReviewTitle = firstReview.textContent.trim();
        return firstReviewTitle !== window.lastReviewTitle;
    }"""

    # --- UPDATED: Extraction Schema with 'review_detail' field ---
    schema = {
        "name": "Reviews",
        "baseSelector": "article.elv-bg-neutral-0",
        "fields": [
            {"name": "title", "selector": 'div[itemprop="name"] div', "type": "text"},
            {
                "name": "author",
                "selector": 'div[itemprop="author"] div',
                "type": "text",
            },
            {
                "name": "review_date",
                "selector": "span label.elv-font-medium",
                "type": "text",
            },
            {
                "name": "overall_rating",
                "selector": "label.elv-font-semibold",
                "type": "text",
            },
            {
                "name": "pros",
                "selector": 'div:contains("What do you like best") + p',
                "type": "text",
            },
            {
                "name": "cons",
                "selector": 'div:contains("What do you dislike") + p',
                "type": "text",
            },
            {
                # ADDED: The new field for review detail
                "name": "review_detail",
                "selector": 'div:contains("What problems is Jira solving") + p',
                "type": "text",
            },
        ],
    }
    extraction_strategy = JsonCssExtractionStrategy(schema)

    # --- Browser Config (Unchanged) ---
    browser_config = BrowserConfig(
        headless=False,  # Set back to True for production runs
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        while True:  # Loop until the last page is reached
            print(f"\n--- Starting scrape for Page {page_number} ---")

            crawler_config = CrawlerRunConfig(
                session_id=session_id,
                extraction_strategy=extraction_strategy,
                js_code=js_next_page_and_save_state if page_number > 1 else None,
                wait_for=(
                    wait_for_new_reviews
                    if page_number > 1
                    else "css:article.elv-bg-neutral-0"
                ),
                js_only=True if page_number > 1 else False,
                cache_mode=CacheMode.BYPASS,
                page_timeout=90000,
                wait_for_timeout=20000,  # More reasonable timeout
            )

            result_container = await crawler.arun(url=url, config=crawler_config)
            result = result_container[0]

            if not (result.success and result.extracted_content):
                print(
                    "❌ Scrape failed or timed out waiting for new content. Likely the last page. Stopping."
                )
                break

            reviews = json.loads(result.extracted_content)

            if not reviews:
                print("✅ Last page reached. No more reviews found.")
                break

            all_reviews.extend(reviews)
            print(f"✅ Page {page_number}: Found {len(reviews)} reviews")
            save_data(reviews, filename=f"g2_jira_reviews_data/page_{page_number}.json")

            page_number += 1

        print(
            f"\n🎉 Scraping complete. Successfully crawled {len(all_reviews)} reviews across {page_number - 1} pages."
        )
        await crawler.crawler_strategy.kill_session(session_id)
        save_data(all_reviews, filename="g2_jira_reviews_data/all_reviews.json")


if __name__ == "__main__":
    asyncio.run(scrape_g2_reviews())
