# Multimodal RAG Ingestion Pipeline

An automated data ingestion pipeline designed to scrape, extract, structure, and index multimodal vehicle listings (text and image galleries) from Otomoto into a Qdrant vector database.

### Key Features & Technical Highlights
* **Web Scraping & Custom Dataset Loading:** Implements resilient scraping routines for Otomoto vehicle pages. Utilizes a custom PyTorch/Python Dataset with **lazy loading** to efficiently handle large volumes of high-resolution images and metadata without exceeding memory bounds.
* **LLM-Powered Data Structuring:** Leverages local LLM inference using **`Qwen/Qwen2.5-3B-Instruct`** to parse, clean, and convert unstructured HTML text and vehicle specs into strictly validated JSON schemas.
* **Image Denoising & Compression Embedder:** Routes all scraped listing images through a custom compression embedder module that filters out visual noise (e.g., watermark artefacts, repetitive backgrounds, minor lighting shifts) while preserving core visual features.
* **Qdrant Vector Database Indexing:** Upserts high-density visual vectors alongside structured text payload attributes into a Qdrant collection for downstream hybrid retrieval.
* **Dynamic Scraper Maintenance Note:** Due to frequent DOM layout updates on the Otomoto platform, the scraping module is designed modularly, allowing selectors and parsing logic to be easily updated without refactoring the core ingestion engine.
