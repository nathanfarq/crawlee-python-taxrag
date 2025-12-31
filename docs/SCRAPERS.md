# Tax Scraper Documentation

This document provides an overview of all available scrapers, their purposes, schedules, and how to run or create new scrapers.

---

## Available Scrapers

### 1. Base Tax Document Scraper
**File:** `tax_rag_project/src/tax_rag_scraper/main.py`
**Workflow:** [.github/workflows/base-scrape-workflow.yml](../.github/workflows/base-scrape-workflow.yml)
**Schedule:** Daily at 9 AM UTC
**Purpose:** Lightweight daily crawl of CRA forms and publications

**Configuration:**
- Max Requests: 100
- Max Depth: 2
- Concurrency: 3

**Start URLs:**
- https://www.canada.ca/en/revenue-agency/services/forms-publications.html

**Run Locally:**
```bash
# Standard daily crawl
python tax_rag_project/src/tax_rag_scraper/main.py

# Deep crawl mode (500 requests, depth 3)
python tax_rag_project/src/tax_rag_scraper/main.py --deep

# Custom depth
python tax_rag_project/src/tax_rag_scraper/main.py --max-depth 4
```

---

### 2. Monthly ITA Scraper
**File:** [src/tax_scraper/scrapers/monthly_ita_scraper.py](../src/tax_scraper/scrapers/monthly_ita_scraper.py)
**Workflow:** [.github/workflows/monthly-ita-scraper.yml](../.github/workflows/monthly-ita-scraper.yml)
**Schedule:** 1st of every month at 2 AM UTC
**Purpose:** Comprehensive crawl of Income Tax Act and regulations

**Configuration:**
- Max Requests: 2000
- Max Depth: 5
- Concurrency: 3

**Start URLs:**
- https://www.canada.ca/en/revenue-agency/services/tax/income-tax-act-consolidated-statutes.html
- https://www.canada.ca/en/revenue-agency/services/tax/technical/income-tax-regulations.html
- https://www.canada.ca/en/revenue-agency/services/forms-publications/publications.html

**Excluded Patterns:**
- `FullText.html` - Skip full text pages
- `.pdf` - Skip PDF files
- `.xml` - Skip XML files
- `PITIndex.html` - Skip index pages
- `/archive/` - Skip archived content
- `/archived-` - Skip archived content

**Run Locally:**
```bash
python -m tax_scraper.scrapers.monthly_ita_scraper
```

---

### 3. Monthly CRA Scraper
**File:** [src/tax_scraper/scrapers/monthly_cra_scraper.py](../src/tax_scraper/scrapers/monthly_cra_scraper.py)
**Workflow:** [.github/workflows/monthly-cra-scraper.yml](../.github/workflows/monthly-cra-scraper.yml)
**Schedule:** 1st of every month at 2 AM UTC
**Purpose:** Comprehensive crawl of CRA forms, guides, and publications
**Status:** ⚠️ Configuration Pending

**Configuration:**
- Max Requests: 1000
- Max Depth: 3
- Concurrency: 3

**Start URLs:**
- ⚠️ TODO: Add CRA start URLs for forms, guides, and publications

**Excluded Patterns:**
- ⚠️ TODO: Add patterns to exclude (PDFs, XML, archived content, etc.)

**Run Locally:**
```bash
python -m tax_scraper.scrapers.monthly_cra_scraper
```

**Note:** This scraper requires configuration before use. Update the `START_URLS`, `ALLOWED_DOMAINS`, and `EXCLUDED_PATTERNS` in the scraper file.

---

### 4. Monthly DoF Scraper
**File:** [src/tax_scraper/scrapers/monthly_dof_scraper.py](../src/tax_scraper/scrapers/monthly_dof_scraper.py)
**Workflow:** [.github/workflows/monthly-dof-scraper.yml](../.github/workflows/monthly-dof-scraper.yml)
**Schedule:** 1st of every month at 2 AM UTC
**Purpose:** Crawl Department of Finance budgets and draft legislation
**Status:** ⚠️ Configuration Pending

**Configuration:**
- Max Requests: 1000
- Max Depth: 3
- Concurrency: 3

**Start URLs:**
- ⚠️ TODO: Add Department of Finance start URLs for budgets and draft legislation

**Excluded Patterns:**
- ⚠️ TODO: Add patterns to exclude (PDFs, XML, archived content, etc.)

**Run Locally:**
```bash
python -m tax_scraper.scrapers.monthly_dof_scraper
```

**Note:** This scraper requires configuration before use. Update the `START_URLS`, `ALLOWED_DOMAINS`, and `EXCLUDED_PATTERNS` in the scraper file.

---

## How to Run Scrapers Locally

### Prerequisites
1. Python 3.11 or higher
2. Install dependencies:
   ```bash
   uv pip install -e ".[tax-rag]"
   playwright install chromium
   ```
3. Set up environment variables in `.env` file:
   ```env
   QDRANT_URL=https://your-cluster.cloud.qdrant.io
   QDRANT_API_KEY=your-api-key
   OPENAI_API_KEY=sk-proj-...
   ```

### Running a Scraper
Each scraper can be run as a standalone Python module:

```bash
# Monthly ITA scraper (configured)
python -m tax_scraper.scrapers.monthly_ita_scraper

# Monthly CRA scraper (needs configuration)
python -m tax_scraper.scrapers.monthly_cra_scraper

# Monthly DoF scraper (needs configuration)
python -m tax_scraper.scrapers.monthly_dof_scraper

# Base scraper (from main.py)
python tax_rag_project/src/tax_rag_scraper/main.py
```

---

## Creating a New Scraper

Follow these steps to create a new scraper based on the monthly ITA template:

### Step 1: Copy the Template
```bash
cp src/tax_scraper/scrapers/monthly_ita_scraper.py src/tax_scraper/scrapers/your_scraper_name.py
```

### Step 2: Update Configuration
Edit the new file and update these sections:

```python
# ==============================================================================
# SCRAPER CONFIGURATION
# ==============================================================================

# Start URLs for your scraper
START_URLS = [
    'https://example.com/your-target-page',
]

# Allowed domains
ALLOWED_DOMAINS = [
    'example.com',
]

# URL patterns to exclude
EXCLUDED_PATTERNS = [
    '.pdf',
    '.xml',
    '/archive/',
]

# Crawl settings
CRAWL_CONFIG = {
    'MAX_DEPTH': 3,              # Adjust based on site structure
    'MAX_CONCURRENCY': 3,        # How many pages to crawl simultaneously
    'MAX_REQUESTS': 500,         # Total page limit
    'CRAWL_TYPE': 'your-type',   # For metrics tracking
}
```

### Step 3: Create a Workflow
Copy the workflow template:

```bash
cp .github/workflows/monthly-ita-scraper.yml .github/workflows/your-scraper-name.yml
```

Update these fields in the new workflow:
- `name:` - Workflow display name
- `schedule:` - Cron expression for when to run
- Job step: `python -m tax_scraper.scrapers.your_scraper_name`
- Environment variables if needed
- Artifact names

**Cron Schedule Examples:**
- Daily: `'0 9 * * *'` (9 AM UTC daily)
- Weekly: `'0 9 * * 1'` (9 AM UTC every Monday)
- Monthly: `'0 2 1 * *'` (2 AM UTC on 1st of month)
- Quarterly: `'0 2 1 1,4,7,10 *'` (Jan, Apr, Jul, Oct)

### Step 4: Test Locally
```bash
python -m tax_scraper.scrapers.your_scraper_name
```

### Step 5: Update Documentation
Add your scraper to this file under "Available Scrapers" section.

---

## Scraper Architecture

All scrapers use the shared infrastructure from `tax_rag_scraper`:

### Core Components
- **Base Crawler** (`tax_rag_scraper/crawlers/base_crawler.py`): Handles crawling logic, rate limiting, and storage
- **Handlers** (`tax_rag_scraper/handlers/`): Extract data from specific sites
- **Storage** (`tax_rag_scraper/storage/`): Qdrant Cloud integration for vector storage
- **Utilities** (`tax_rag_scraper/utils/`): Embeddings, link extraction, robots.txt checking

### Scraper Files
Each scraper file should be self-contained with:
1. **Configuration**: URLs, domains, exclusion patterns, crawl settings
2. **Main Function**: Validates credentials and runs the crawler
3. **Shared Infrastructure**: Uses `TaxDataCrawler` from base module

### Benefits of This Architecture
- **Reusability**: Share common crawling logic
- **Maintainability**: Update one file to fix all scrapers
- **Simplicity**: Each scraper is just configuration + a main function
- **Testability**: Easy to run locally with different settings

---

## Environment Variables

### Required
- `QDRANT_URL` - Qdrant Cloud cluster URL
- `QDRANT_API_KEY` - Qdrant Cloud API key
- `OPENAI_API_KEY` - OpenAI API key for embeddings

### Optional (with defaults)
- `MAX_REQUESTS_PER_CRAWL` - Default: varies by scraper
- `MAX_CONCURRENCY` - Default: 3
- `MAX_CRAWL_DEPTH` - Default: varies by scraper
- `QDRANT_COLLECTION` - Default: `tax_documents`
- `USE_QDRANT` - Default: `true`
- `STORAGE_DIR` - Default: `./tax_rag_project/storage`
- `MAX_REQUESTS_PER_MINUTE` - Default: 60
- `MIN_REQUEST_DELAY` - Default: 1.0
- `MAX_REQUEST_DELAY` - Default: 3.0

---

## Monitoring and Metrics

### GitHub Actions Artifacts
Each workflow uploads two artifacts:
1. **Metrics** (`metrics.jsonl`) - Crawl statistics in JSONL format
2. **Results** - Full crawl data and stored documents

### Metrics Format
```json
{
  "timestamp": "2025-01-15T09:00:00Z",
  "crawl_type": "monthly-ita",
  "urls_processed": 1523,
  "successful_requests": 1498,
  "failed_requests": 25,
  "success_rate": 98.36,
  "duration_seconds": 1847.3
}
```

### Viewing Results
- **GitHub Actions**: Check workflow runs for artifacts
- **Qdrant Cloud**: https://cloud.qdrant.io - View stored documents
- **Local Storage**: `tax_rag_project/storage/datasets/`

---

## Troubleshooting

### Common Issues

**Issue: "QDRANT_URL environment variable not set"**
- Solution: Add credentials to `.env` file or GitHub Secrets

**Issue: Scraper runs but no results**
- Check if URLs are accessible
- Verify robots.txt allows crawling
- Check logs for error messages

**Issue: Rate limiting errors**
- Reduce `MAX_CONCURRENCY`
- Increase `MIN_REQUEST_DELAY`
- Reduce `MAX_REQUESTS_PER_MINUTE`

**Issue: Module not found errors**
- Run: `uv pip install --system -e ".[tax-rag]"`
- Ensure you're in the project root directory

---

## Best Practices

1. **Start Small**: Test with low `MAX_REQUESTS` first
2. **Respect robots.txt**: Always enabled by default
3. **Be Polite**: Use appropriate delays and concurrency
4. **Monitor Costs**: OpenAI embeddings and Qdrant storage have costs
5. **Version Control**: Commit scraper changes before running in production
6. **Test Locally**: Always test new scrapers locally before deploying

---

## Additional Resources

- [Crawlee Documentation](https://crawlee.dev/python/)
- [Qdrant Cloud](https://cloud.qdrant.io)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [GitHub Actions Cron Syntax](https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows#schedule)

---

**Last Updated:** 2025-12-29
