# Production Equity Research Platform

A comprehensive, production-ready equity research platform with institutional-quality analytics, multi-source data integration, and microservices architecture.

## 🎯 Features

### **Analytics Suite**
- **Financial Modeling**: DCF valuation with sensitivity analysis
- **Quant Signals**: Momentum, RSI, SMA crossovers, ATR
- **Peer Analysis**: Industry benchmarking with percentiles
- **Macro Analysis**: FRED integration with regime detection
- **Report Generation**: Automated research notes in Markdown

### **Data Sources**
- **Financial Modeling Prep**: Fundamentals, prices, enterprise values
- **FRED**: Macro indicators (CPI, rates, spreads)
- **SEC EDGAR**: 10-K/10-Q filings with vector search
- **Provider Layer**: Clean abstractions for easy extension

### **Production Features**
- **Microservices**: 6 specialized FastAPI services
- **Vector Search**: Qdrant with OpenAI embeddings
- **Caching**: Redis with intelligent TTLs
- **Monitoring**: Prometheus metrics and health checks
- **Security**: Rate limiting and error handling

## 🚀 Quick Start

### 1. Infrastructure
```bash
docker compose -f infra/docker-compose.yml up -d
```

### 2. Dependencies
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
```

### 3. Environment
```bash
cp .env.example .env
# Edit .env with your API keys:
# - OPENAI_API_KEY=sk-...
# - FMP_API_KEY=...
# - SEC_USER_AGENT="YourApp ([email protected])"
# - FRED_API_KEY=... (optional)
```

### 4. Optional: Ingest Data
```bash
python ingestion/edgar_ingest_minimal.py --ticker AAPL --limit 1
```

### 5. Start Services
```bash
# Terminal 1: Retriever
uvicorn retriever.service:app --host 0.0.0.0 --port 8081

# Terminal 2: Models
uvicorn models.service:app --host 0.0.0.0 --port 8082

# Terminal 3: Macro
uvicorn apis.macro.service:app --host 0.0.0.0 --port 8083

# Terminal 4: Quant
uvicorn apis.quant.service:app --host 0.0.0.0 --port 8084

# Terminal 5: Comps
uvicorn apis.comps.service:app --host 0.0.0.0 --port 8085

# Terminal 6: Reports
uvicorn apis.reports.service:app --host 0.0.0.0 --port 8086
```

## 🧪 Testing

### Health Checks
```bash
curl "http://localhost:8081/v1/health"  # Retriever
curl "http://localhost:8082/v1/health"  # Models
curl "http://localhost:8083/v1/health"  # Macro
curl "http://localhost:8084/v1/health"  # Quant
curl "http://localhost:8085/v1/health"  # Comps
curl "http://localhost:8086/v1/health"  # Reports
```

### API Examples
```bash
# Financial model with DCF
curl "http://localhost:8082/v1/model/AAPL"

# Vector search (requires ingested data)
curl "http://localhost:8081/v1/retrieve?query=gross%20margin&tickers=AAPL&k=5"

# Macro snapshot
curl "http://localhost:8083/v1/macro"

# Quant signals
curl "http://localhost:8084/v1/quant/AAPL"

# Peer comparisons
curl "http://localhost:8085/v1/comps/AAPL"

# Complete research report
curl "http://localhost:8086/v1/report/AAPL"
```

## 📁 Architecture

```
production_equity_research_platform/
├── src/services/           # Core business logic
│   ├── providers/         # Data source abstractions
│   ├── cache/            # Redis caching
│   ├── monitoring/       # Metrics and health
│   ├── quant/           # Technical analysis
│   ├── comps/           # Peer analysis
│   ├── macro/           # Economic indicators
│   └── report/          # Report generation
├── models/              # Financial modeling service
├── retriever/           # Vector search service
├── apis/               # Specialized API services
│   ├── macro/          # Macro analysis API
│   ├── quant/          # Quant signals API
│   ├── comps/          # Peer comparison API
│   └── reports/        # Report generation API
├── ingestion/          # EDGAR data pipeline
└── infra/             # Docker infrastructure
```

## 🔧 Configuration

### Ranking Weights (retriever/config.yaml)
```yaml
ranking:
  alpha: 0.4   # semantic similarity
  beta: 0.3    # recency weight
  gamma: 0.2   # source type priority
  delta: 0.1   # ticker match
```

### Environment Variables
- `OPENAI_API_KEY`: Required for embeddings
- `FMP_API_KEY`: Required for financial data
- `FRED_API_KEY`: Optional for macro data
- `SEC_USER_AGENT`: Required for EDGAR ingestion
- `QDRANT_URL`: Vector database URL
- `REDIS_URL`: Cache database URL

## 📊 Data Coverage

### Financial Modeling Prep
- Income statements, balance sheets, cash flows
- Key metrics, enterprise values, stock prices
- Peer screening by sector/industry
- 8+ years of historical data

### FRED (Federal Reserve)
- CPI, interest rates, credit spreads
- Economic indicators for regime detection
- Historical time series data

### SEC EDGAR
- 10-K/10-Q filings with full text search
- Automated chunking and embedding
- Citation-ready metadata

## 🎯 Use Cases

### **Investment Research**
- Generate institutional-quality research reports
- Perform comprehensive peer analysis
- Track macro regime changes

### **Quantitative Analysis**
- Technical signal generation
- Multi-factor ranking systems
- Historical backtesting data

### **Compliance & Documentation**
- Source-attributed analysis
- Audit trail for all calculations
- Regulatory-compliant disclaimers

## 🔄 Extending the Platform

### Adding New Data Providers
1. Implement provider interface in `src/services/providers/`
2. Add caching and error handling
3. Update environment configuration

### Creating New Analytics
1. Add service in `src/services/analytics/`
2. Create FastAPI service in `apis/`
3. Update report composer

### Scaling Deployment
- Use Docker Swarm or Kubernetes
- Add load balancers for API services
- Implement distributed caching

## 📈 Performance

- **Sub-second response times** for cached data
- **Intelligent caching** with appropriate TTLs
- **Batch processing** for bulk operations
- **Connection pooling** for database efficiency

## 🛡️ Security

- **Rate limiting** on all endpoints
- **Input validation** and sanitization
- **Error handling** without information leakage
- **Environment-based secrets** management

---

**Ready for production deployment and institutional-quality equity research.**

