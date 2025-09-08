#!/bin/bash

# AI Equity Research Platform - Post-Fix Smoke Tests
# Run this script after applying fixes to validate system integrity

set -e

echo "🚀 AI Equity Research Platform - Post-Fix Smoke Tests"
echo "======================================================"

# Check if we're in the right directory
if [ ! -f "smoke_tests.py" ]; then
    echo "❌ Error: smoke_tests.py not found. Please run from the project root directory."
    exit 1
fi

# Make sure Python dependencies are available
echo "📦 Checking Python environment..."
python3 -c "import sys; print(f'Python: {sys.version}')"

# Install test dependencies if needed
if ! python3 -c "import pytest" 2>/dev/null; then
    echo "Installing pytest..."
    pip install pytest pytest-asyncio
fi

if ! python3 -c "import requests" 2>/dev/null; then
    echo "Installing requests..."
    pip install requests
fi

echo ""

# Run the comprehensive smoke tests
echo "🧪 Running comprehensive smoke tests..."
python3 smoke_tests.py

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "🎉 All smoke tests completed successfully!"
    echo ""
    echo "Manual checks to perform:"
    echo "1. Start the service: uvicorn apis.reports.service:app --host 0.0.0.0 --port 8086"
    echo "2. Test metrics endpoint: curl localhost:8086/metrics"
    echo "3. Test report endpoint: curl localhost:8086/v1/report/AAPL"
    echo ""
else
    echo ""
    echo "⚠️  Some smoke tests failed. Please review the output above."
    echo ""
fi

exit $exit_code

