FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Compile-time syntax check
RUN python -m py_compile ui_minimal.py \
 && python -m py_compile src/services/providers/fmp_provider.py \
 && python -m py_compile src/services/financial_modeler.py

# Start app
CMD ["uvicorn", "ui_minimal:app", "--host", "0.0.0.0", "--port", "8080"]
