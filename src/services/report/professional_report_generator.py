from src.services.report.professional_report_generator import professional_report_generator

@app.get("/pro-report", response_class=HTMLResponse)
def pro_report(ticker: str = Query(...)):
    model = build_model(ticker, force_refresh=False)
    payload = professional_report_generator.make_payload_from_model(ticker, model, quant=None)
    html, charts = professional_report_generator.generate_report_with_charts(payload)
    panel = f"""
    <div class="panel">
      <h2>{ticker.upper()} Professional Report</h2>
      <div style="margin-bottom:12px;">
        <a href="/report.md?ticker={ticker}" class="btn" style="text-decoration:none;">Download Markdown</a>
      </div>
      <div style="border:1px solid var(--line);padding:18px;border-radius:12px;">{html}</div>
    </div>
    """
    return HTMLResponse(render(panel))
