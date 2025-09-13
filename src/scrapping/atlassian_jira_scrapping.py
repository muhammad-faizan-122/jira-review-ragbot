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


async def scrape_jira_reviews():
    url = "https://www.softwareadvice.com/project-management/atlassian-jira-profile/reviews/"
    session_id = "software_advice_full_scrape_session"
    all_reviews = []
    page_number = 1

    # --- JavaScript Snippets for State Management and Navigation ---

    # JS to save the current state (first review title) and click the 'Next' button.
    js_next_page_and_save_state = """
    const firstReview = document.querySelector('div[data-testid="text-review-card"] p.text-2xl');
    if (firstReview) {
        window.lastReviewTitle = firstReview.textContent.trim();
    }
    const button = document.querySelector('button[data-testid="next-button"]');
    // Click only if the button exists and is not disabled
    if (button && !button.disabled) { button.click(); }
    """

    # JS function to wait for the page update. It's the most reliable way to
    # confirm new content has loaded. The crawler will timeout if this condition is not met.
    wait_for_new_reviews = """
    () => {
        const firstReview = document.querySelector('div[data-testid="text-review-card"] p.text-2xl');
        if (!firstReview) return false;
        const firstReviewTitle = firstReview.textContent.trim();
        // Return true only when the new title is different from the one we saved
        return firstReviewTitle !== window.lastReviewTitle;
    }"""

    # --- Extraction Schema ---
    schema = {
        "name": "Reviews",
        "baseSelector": 'div[data-testid="text-review-card"]',
        "fields": [
            {"name": "title", "selector": "p.text-2xl", "type": "text"},
            {
                "name": "author",
                "selector": 'p[data-testid="reviewer-first-name"]',
                "type": "text",
            },
            {
                "name": "review_date",
                "selector": 'p[data-testid="reviewed-date"]',
                "type": "text",
            },
            {
                "name": "overall_rating",
                "selector": 'p[data-testid="review-overall-rating-value"]',
                "type": "text",
            },
            {
                "name": "pros",
                "selector": 'p.font-bold:contains("Pros") + p',
                "type": "text",
            },
            {
                "name": "cons",
                "selector": 'p.font-bold:contains("Cons") + p',
                "type": "text",
            },
            {
                "name": "review_detail",
                "selector": "p.text-sm text-grey-91 + p",
                "type": "text",
            },
        ],
    }
    extraction_strategy = JsonCssExtractionStrategy(schema)

    # --- Browser Config ---
    browser_config = BrowserConfig(
        headless=True,  # Set to False to watch the process
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        while True:  # Loop indefinitely until we decide to break
            print(f"\n--- Starting scrape for Page {page_number} ---")

            # --- Dynamically create the config for the current loop iteration ---
            crawler_config = CrawlerRunConfig(
                session_id=session_id,
                extraction_strategy=extraction_strategy,
                # Use the 'click and wait' logic for all pages AFTER the first one.
                js_code=js_next_page_and_save_state if page_number > 1 else None,
                wait_for=(
                    wait_for_new_reviews
                    if page_number > 1
                    else "css:div[data-testid='text-review-card']"
                ),
                js_only=True if page_number > 1 else False,
                cache_mode=CacheMode.BYPASS,
                page_timeout=60000,  # Allow ample time for pages to load
                wait_for_timeout=20000,  # Timeout for the wait_for condition
            )

            # --- Run the crawler ---
            result_container = await crawler.arun(url=url, config=crawler_config)
            result = result_container[0]

            # --- Check the result and decide whether to continue ---
            if not (result.success and result.extracted_content):
                # This will trigger if the 'wait_for' times out on the last page, which is expected.
                print(
                    "❌ Scrape failed or timed out waiting for new content. Likely the last page. Stopping."
                )
                break

            reviews = json.loads(result.extracted_content)

            # The most reliable end condition: extraction succeeded but found nothing.
            if not reviews:
                print("✅ Last page reached. No more reviews found.")
                break

            all_reviews.extend(reviews)
            print(f"✅ Page {page_number}: Found {len(reviews)} reviews")
            save_data(reviews, filename=f"scrapped_data/page_{page_number}.json")

            page_number += 1  # Increment for the next loop

        print(
            f"\n🎉 Scraping complete. Successfully crawled {len(all_reviews)} reviews across {page_number - 1} pages."
        )
        await crawler.crawler_strategy.kill_session(session_id)
        # save_data(all_reviews, filename="atlassian_jira_reviews_data/all_reviews.json")


if __name__ == "__main__":
    asyncio.run(scrape_jira_reviews())
