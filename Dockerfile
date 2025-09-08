# Dockerfile  (for the API service)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ---- deps ----
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---- app code ----
COPY . .

# ---- quick syntax checks (safe: doesn't import/execute) ----
RUN python -m py_compile \
      apis/reports/service.py \
      src/services/report/professional_report_generator.py \
      src/services/providers/fmp_provider.py \
      src/services/financial_modeler.py

# (optional) document port; Render will inject $PORT
EXPOSE 8080

# ---- start the Reports API ----
# Use sh -c so ${PORT} expands on Render
CMD ["sh","-c","uvicorn apis.reports.service:app --host 0.0.0.0 --port ${PORT:-8080}"]
