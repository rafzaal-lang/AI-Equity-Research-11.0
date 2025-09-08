#!/usr/bin/env python3
import os, sys, subprocess, json, shutil
from pathlib import Path

ROOT = Path(__file__).parent

def run(cmd, check=True):
    print(f"$ {' '.join(cmd)}")
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0 and check:
        print(p.stdout); print(p.stderr)
        raise SystemExit(p.returncode)
    return p

def test_python_compile():
    files = [str(p) for p in ROOT.rglob("*.py") if ".venv" not in str(p)]
    cmd = [sys.executable, "-m", "py_compile", *files]
    run(cmd)

def test_requirements():
    req = (ROOT / "requirements.txt").read_text()
    assert "..." not in req, "requirements.txt contains '...' placeholder"
    for must in ["tenacity", "seaborn"]:
        assert must in req, f"Missing dependency in requirements: {must}"

def test_reports_health():
    # optional: only if service is running locally
    import urllib.request, json
    try:
        with urllib.request.urlopen("http://127.0.0.1:8086/v1/health", timeout=2) as r:
            data = json.loads(r.read().decode())
            assert data.get("status") == "healthy"
    except Exception as e:
        print(f"[warn] reports service /v1/health not reachable: {e}")

def main():
    print("== Smoke Tests ==")
    test_python_compile()
    test_requirements()
    test_reports_health()
    print("All smoke tests completed.")

if __name__ == "__main__":
    main()

