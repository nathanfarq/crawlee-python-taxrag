"""ITA scraper for Canadian Income Tax Act and regulations.

This scraper crawls Income Tax Act and regulations.
It's designed to be self-contained and easily replicable for other scrapers.

Usage:
    python -m tax_rag_scraper.active-crawlers.ita_scraper
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Import from tax_rag_scraper (the actual module)
from tax_rag_scraper.config.settings import Settings
from tax_rag_scraper.crawlers.base_crawler import TaxDataCrawler

logger = logging.getLogger(__name__)

# ==============================================================================
# SCRAPER CONFIGURATION
# ==============================================================================

# Start URLs for ITA crawl
START_URLS = [
    'https://laws-lois.justice.gc.ca/eng/acts/I-3.3/',
    'https://laws-lois.justice.gc.ca/eng/regulations/c.r.C.,_c._945/index.html',
]

# Allowed domains for this scraper
ALLOWED_DOMAINS = [
    'https://laws-lois.justice.gc.ca/eng/acts/I-3.3/**',
    'https://laws-lois.justice.gc.ca/eng/regulations/c.r.C.,_c._945/**',
]

# URL patterns to exclude (PDFs, XML, FullText pages, etc.)
EXCLUDED_PATTERNS = [
    'https://laws-lois.justice.gc.ca/eng/acts/I-3.3/FullText.html',
    'https://laws-lois.justice.gc.ca/eng/regulations/c.r.C.,_c._945/FullText.html',
    'https://laws-lois.justice.gc.ca/**.xml',
    'https://laws-lois.justice.gc.ca/**.pdf',
    'https://laws-lois.justice.gc.ca/**/PITIndex.html',
]

# Crawl settings for ITA scraper
CRAWL_CONFIG = {
    'MAX_DEPTH': 5,
    'MAX_CONCURRENCY': 3,
    'MAX_REQUESTS': 5000,
    'CRAWL_TYPE': 'ita',
    'COLLECTION': 'ita-collection',
    'SOURCE': 'ita',
}

# ==============================================================================
# MAIN SCRAPER FUNCTION
# ==============================================================================


async def main() -> None:
    """Run the ITA scraper."""
    # Load environment variables
    env_path = Path(__file__).parent.parent.parent.parent / 'tax_rag_project' / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        logger.info('[OK] Loaded environment from .env')
    else:
        logger.warning('[WARNING] .env file not found, using environment variables')

    # Validate required credentials
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')
    openai_api_key = os.getenv('OPENAI_API_KEY')

    if not qdrant_url:
        logger.error('\n[ERROR] QDRANT_URL environment variable not set')
        logger.error('\nTo use Qdrant Cloud:')
        logger.error('  1. Visit https://cloud.qdrant.io')
        logger.error('  2. Create a free account (1GB storage included)')
        logger.error('  3. Create a new cluster')
        logger.error('  4. Copy your cluster URL')
        logger.error('  5. Add to .env file or GitHub Secrets:')
        logger.error('     QDRANT_URL=https://your-cluster.cloud.qdrant.io')
        logger.error('     QDRANT_API_KEY=your-api-key')
        sys.exit(1)

    if not qdrant_api_key:
        logger.error('\n[ERROR] QDRANT_API_KEY environment variable not set')
        logger.error('\nGet your API key from https://cloud.qdrant.io')
        logger.error('Add to .env file or GitHub Secrets: QDRANT_API_KEY=your-api-key')
        sys.exit(1)

    if not openai_api_key:
        logger.error('\n[ERROR] OPENAI_API_KEY environment variable not set')
        logger.error('\nGet your API key from https://platform.openai.com/api-keys')
        logger.error('Add to .env file or GitHub Secrets: OPENAI_API_KEY=sk-proj-...')
        sys.exit(1)

    logger.info('[OK] Qdrant Cloud URL: %s', qdrant_url)
    logger.info('[OK] OpenAI API key configured')

    # Configure settings with scraper overrides
    settings = Settings()
    settings.MAX_CRAWL_DEPTH = CRAWL_CONFIG['MAX_DEPTH']
    settings.MAX_CONCURRENCY = CRAWL_CONFIG['MAX_CONCURRENCY']
    settings.MAX_REQUESTS_PER_CRAWL = CRAWL_CONFIG['MAX_REQUESTS']
    settings.QDRANT_COLLECTION = CRAWL_CONFIG['COLLECTION']
    settings.QDRANT_SOURCE = CRAWL_CONFIG['SOURCE']

    logger.info('\n[INFO] Starting ITA Scraper')
    logger.info('[INFO] Target: Canadian Income Tax Act & Regulations')
    logger.info('[INFO] Collection: %s', settings.QDRANT_COLLECTION)
    logger.info('[INFO] Max requests: %d', settings.MAX_REQUESTS_PER_CRAWL)
    logger.info('[INFO] Max depth: %d', settings.MAX_CRAWL_DEPTH)
    logger.info('[INFO] Concurrency: %d', settings.MAX_CONCURRENCY)
    logger.info('[INFO] Start URLs:')
    for url in START_URLS:
        logger.info('  - %s', url)
    logger.info('')

    # Create crawler with Qdrant Cloud integration
    crawler = TaxDataCrawler(
        settings=settings,
        max_depth=CRAWL_CONFIG['MAX_DEPTH'],
        use_qdrant=settings.USE_QDRANT,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
    )

    # Run the crawler
    await crawler.run(START_URLS, crawl_type=CRAWL_CONFIG['CRAWL_TYPE'])

    logger.info('\n[OK] ITA scraper complete.')
    logger.info('[OK] Check storage/datasets/default/ for results.')
    logger.info('[OK] View your data in Qdrant Cloud: https://cloud.qdrant.io')


if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    # Run the scraper
    asyncio.run(main())
