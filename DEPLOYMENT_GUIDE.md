# Deployment Guide: AI Equity Research Platform

This guide provides instructions for deploying the AI Equity Research Platform.

## 1. Prerequisites

- Docker and Docker Compose
- Python 3.8+
- Redis
- An account with Financial Modeling Prep (FMP) for an API key

## 2. Configuration

1.  **Environment Variables:** Create a `.env` file in the root directory and populate it with the following:

    ```
    FMP_API_KEY=your_fmp_api_key
    REDIS_HOST=localhost
    REDIS_PORT=6379
    LOG_LEVEL=INFO
    ```

## 3. Running with Docker

```bash
# Build and run the services
docker-compose up --build
```

## 4. Running Locally

1.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Start Services:**

    ```bash
    # Run the main application (assuming a Flask app in main.py)
    python main.py
    ```

## 5. Health Checks

-   **Liveness:** `GET /health/live`
-   **Readiness:** `GET /health/ready`
-   **Full Health Check:** `GET /health/full`


