# scraper/amazon_scraper.py
import scrapy
import json
import re
from datetime import datetime
from scrapy.crawler import CrawlerProcess

class AmazonReviewSpider(scrapy.Spider):
    name = "amazon_reviews"
    custom_settings = {
        "USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "ROBOTSTXT_OBEY": False,
        "DOWNLOAD_DELAY": 2,
        "RANDOMIZE_DOWNLOAD_DELAY": True,
        "AUTOTHROTTLE_ENABLED": True,
        "FEEDS": {"raw_reviews.json": {"format": "json"}},
    }

    def __init__(self, asin="B08N5WRWNW", pages=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.asin = asin
        self.pages = int(pages)
        self.base_url = (
            f"https://www.amazon.com/product-reviews/{asin}"
            "/?reviewerType=all_reviews&pageNumber="
        )

    def start_requests(self):
        for page in range(1, self.pages + 1):
            yield scrapy.Request(
                url=self.base_url + str(page),
                callback=self.parse,
                meta={"page": page},
            )

    def parse(self, response):
        for review in response.css("div[data-hook='review']"):
            yield {
                "review_id": review.attrib.get("id"),
                "product_asin": self.asin,
                "reviewer_id": self._extract_reviewer_id(review),
                "reviewer_name": review.css(
                    "span.a-profile-name::text"
                ).get("").strip(),
                "rating": float(
                    review.css("i[data-hook='review-star-rating'] span::text")
                    .get("0 out").split()[0]
                ),
                "title": review.css(
                    "a[data-hook='review-title'] span:last-child::text"
                ).get("").strip(),
                "body": review.css(
                    "span[data-hook='review-body'] span::text"
                ).get("").strip(),
                "date": review.css(
                    "span[data-hook='review-date']::text"
                ).get("").strip(),
                "verified_purchase": bool(
                    review.css("span[data-hook='avp-badge']").get()
                ),
                "helpful_votes": self._parse_helpful(review),
                "scraped_at": datetime.utcnow().isoformat(),
            }

    def _extract_reviewer_id(self, review):
        link = review.css("a.a-profile::attr(href)").get("")
        match = re.search(r"/profile/([A-Z0-9]+)", link)
        return match.group(1) if match else "UNKNOWN"

    def _parse_helpful(self, review):
        text = review.css("span[data-hook='helpful-vote-statement']::text").get("")
        match = re.search(r"(\d+)", text)
        return int(match.group(1)) if match else 0


if __name__ == "__main__":
    process = CrawlerProcess()
    process.crawl(AmazonReviewSpider, asin="B08N5WRWNW", pages=10)
    process.start()