# Post-Fix Smoke Test Guide

This guide outlines the smoke tests that should be run after applying fixes to ensure the AI Equity Research Platform works correctly.

## Quick Start

```bash
# Run all automated smoke tests
./run_smoke_tests.sh

# Or run individual tests
python3 smoke_tests.py
```

## Test Coverage

### 1. Syntax Compile Check ✅
**Command:** `python -m py_compile $(git ls-files '*.py')`

**What it tests:**
- All Python files compile without syntax errors
- Template syntax is valid
- Import statements are correct

**Expected result:** All files compile successfully

### 2. Unit Tests ✅
**Command:** `pytest -q`

**What it tests:**
- Core functionality works as expected
- Mocked provider tests (no network calls)
- Data validation and processing

**Expected result:** All tests pass

### 3. Report EML Round-Trip ✅
**Test:** `ProfessionalReportGenerator.generate_full_report(report_data, output_format="eml")`

**What it validates:**
- EML file is generated successfully
- Subject line: `"Equity Research Report: <Company Name>"`
- Required Content-IDs present:
  - `<profitability_chart>`
  - `<liquidity_chart>`
- Optional Content-IDs (when data provided):
  - `<dcf_sensitivity_chart>`
  - `<comps_chart>`

**Expected result:** EML contains proper structure and embedded charts

### 4. Metrics Endpoint ✅
**Command:** `curl localhost:8086/metrics`

**What it tests:**
- `/metrics` endpoint returns Prometheus format
- Content-Type is correct
- Metrics are properly formatted

**Expected result:** Prometheus text format response

### 5. Redis Client ✅
**What it tests:**
- Redis connection works
- Basic operations (set/get/delete)
- Client interface compatibility
- Error handling

**Expected result:** All Redis operations work correctly

## Manual Verification Steps

After running automated tests, perform these manual checks:

### 1. Start the Service
```bash
uvicorn apis.reports.service:app --host 0.0.0.0 --port 8086
```

### 2. Test API Endpoints
```bash
# Health check
curl localhost:8086/v1/health

# Metrics endpoint
curl localhost:8086/metrics

# Report generation (requires FMP API key)
curl localhost:8086/v1/report/AAPL
```

### 3. Verify Docker Build
```bash
docker build -t equity-research .
docker run -p 8086:8086 equity-research
```

## Common Issues and Fixes

### Syntax Errors
- Check quote escaping in f-strings
- Verify Jinja2 template syntax
- Fix import statements

### Missing Dependencies
- Update requirements.txt
- Install missing packages: `pip install -r requirements.txt`

### Redis Connection Issues
- Ensure Redis is running: `redis-server`
- Check connection settings in environment variables

### Template Rendering Issues
- Verify chart generation works
- Check template variable names match data structure
- Ensure all required data is provided

## Environment Setup

### Required Environment Variables
```bash
FMP_API_KEY=your_fmp_api_key
REDIS_HOST=localhost
REDIS_PORT=6379
LOG_LEVEL=INFO
```

### Optional Environment Variables
```bash
REDIS_PASSWORD=your_redis_password
OPENAI_API_KEY=your_openai_key
```

## Troubleshooting

### Test Failures
1. Check the specific error message
2. Verify all dependencies are installed
3. Ensure environment variables are set
4. Check that external services (Redis) are running

### EML Generation Issues
1. Verify matplotlib/seaborn are installed
2. Check that chart data is properly formatted
3. Ensure template files exist and are readable

### API Endpoint Issues
1. Check that the service starts without errors
2. Verify port 8086 is available
3. Test with curl or browser
4. Check logs for detailed error messages

## Success Criteria

All smoke tests should pass with:
- ✅ No syntax errors in any Python files
- ✅ All unit tests passing
- ✅ EML reports generated with proper structure
- ✅ Metrics endpoint returning Prometheus format
- ✅ Redis client working correctly
- ✅ API endpoints responding correctly
- ✅ Docker build completing successfully

## Reporting Issues

If any smoke tests fail:
1. Note the specific test that failed
2. Copy the full error message
3. Include environment details (Python version, OS, etc.)
4. Provide steps to reproduce the issue

