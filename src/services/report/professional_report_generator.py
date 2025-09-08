# src/services/report/professional_report_generator.py
import os
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
import matplotlib.pyplot as plt

try:
    import seaborn as sns  # optional; we fall back to matplotlib if missing
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False

import pandas as pd

logger = logging.getLogger(__name__)


class ProfessionalReportGenerator:
    """
    Generate professional, visually appealing equity research reports.
    Charts are saved to disk; the caller can inline them (base64) or serve statically.
    """

    def __init__(self):
        template_dir = Path(__file__).parent / "templates"
        self.env = Environment(loader=FileSystemLoader(template_dir), autoescape=True)
        self.report_output_dir = Path(os.getenv("REPORT_OUTPUT_DIR", "reports"))
        self.report_output_dir.mkdir(parents=True, exist_ok=True)

        if _HAS_SNS:
            sns.set_theme(style="whitegrid")

    # --------------------------- chart helpers ---------------------------
    def _new_fig(self, w: int = 10, h: int = 6):
        plt.figure(figsize=(w, h))

    def _save_chart(self) -> Path:
        path = self.report_output_dir / f"{uuid.uuid4().hex[:8]}.png"
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight", dpi=150)
        plt.close()
        return path

    def _line_or_bar(
        self,
        df: pd.DataFrame,
        x: str,
        y_cols: List[str],
        title: str,
        xlabel: str,
        ylabel: str,
        kind: str = "line",
    ) -> Path:
        self._new_fig()
        if kind == "bar":
            for col in y_cols:
                plt.bar(df[x], df[col], label=col)
        else:  # line (default)
            for col in y_cols:
                if _HAS_SNS:
                    sns.lineplot(data=df, x=x, y=col, label=col, marker=".")
                else:
                    plt.plot(df[x], df[col], marker=".", label=col)
        plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel); plt.legend()
        return self._save_chart()

    def _heatmap(self, z: List[List[float]], x_labels: List[str], y_labels: List[str], title: str) -> Path:
        self._new_fig()
        if _HAS_SNS:
            sns.heatmap(pd.DataFrame(z, index=y_labels, columns=x_labels), annot=False)
        else:
            plt.imshow(z, aspect="auto", origin="upper")
            plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha="right")
            plt.yticks(range(len(y_labels)), y_labels)
            plt.colorbar()
        plt.title(title)
        return self._save_chart()

    # --------------------------- chart packer ---------------------------
    def _create_charts(self, financial_data: Dict[str, Any]) -> Dict[str, Path]:
        """
        Create charts from a normalized `financial_data` block:
          financial_data = {
            "periods": [...],                       # optional
            "profitability": {...},                # optional time series dicts
            "liquidity": {...},                    # optional time series dicts
            "valuation": {"sensitivity": {...}},   # optional {"z":..., "x_labels":..., "y_labels":...}
            "comparable_analysis": {"peers":[{"ticker":..., "ev_ebitda":..., "pe":..., "market_cap":...}]}
          }
        """
        charts: Dict[str, Path] = {}

        # Profitability
        if financial_data.get("periods") and isinstance(financial_data.get("profitability"), dict):
            prof = dict(financial_data["profitability"])
            prof["period"] = financial_data["periods"]
            df = pd.DataFrame(prof)
            charts["profitability_chart"] = self._line_or_bar(
                df, "period", [c for c in df.columns if c != "period"], "Profitability Margins", "Period", "Margin", "line"
            )

        # Liquidity / solvency
        if financial_data.get("periods") and isinstance(financial_data.get("liquidity"), dict):
            liq = dict(financial_data["liquidity"])
            liq["period"] = financial_data["periods"]
            df = pd.DataFrame(liq)
            charts["liquidity_chart"] = self._line_or_bar(
                df, "period", [c for c in df.columns if c != "period"], "Liquidity & Solvency", "Period", "Ratio", "line"
            )

        # DCF sensitivity heatmap (expects a prebuilt grid)
        sens = (financial_data.get("valuation") or {}).get("sensitivity") or {}
        if isinstance(sens, dict) and sens.get("z") and sens.get("x_labels") and sens.get("y_labels"):
            charts["dcf_sensitivity_chart"] = self._heatmap(
                sens["z"], sens["x_labels"], sens["y_labels"], "DCF EV Sensitivity (WACC vs Terminal Growth)"
            )

        # Comps chart – prefers EV/EBITDA, falls back to P/E if EV/EBITDA not present
        peers = ((financial_data.get("comparable_analysis") or {}).get("peers")) or []
        if peers:
            df = pd.DataFrame(peers)
            metric = None
            label = None
            if "ev_ebitda" in df.columns and df["ev_ebitda"].notna().any():
                metric = "ev_ebitda"; label = "EV/EBITDA"
            elif "pe" in df.columns and df["pe"].notna().any():
                metric = "pe"; label = "P/E"
            if metric:
                df = df[["ticker", metric]].dropna().sort_values(metric).tail(12)
                self._new_fig()
                plt.bar(df["ticker"], df[metric])
                plt.xticks(rotation=45, ha="right"); plt.ylabel(label); plt.title("Comparable Company Multiples")
                charts["comps_chart"] = self._save_chart()

        return charts

    # --------------------------- HTML / EML renderers ---------------------------
    def generate_html_report(self, report_data: Dict[str, Any], charts: Optional[Dict[str, Path]] = None) -> str:
        """
        Render HTML with the Jinja template. We expose `charts` to the template
        as strings (paths); the caller may inline them later.
        """
        try:
            template = self.env.get_template("equity_research_report.html")
            expanded = dict(report_data)
            expanded["generation_date"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            expanded["report_id"] = uuid.uuid4().hex
            expanded["report_version"] = "1.1"
            expanded["charts"] = {k: str(v) for k, v in (charts or {}).items()}
            return template.render(expanded)
        except Exception as e:
            logger.error("Error generating HTML report: %s", e)
            raise

    def generate_report_with_charts(self, report_data: Dict[str, Any]) -> Tuple[str, Dict[str, Path]]:
        """Create charts and return (html, charts)."""
        charts = self._create_charts(report_data.get("financial_summary", {}) or {})
        html = self.generate_html_report(report_data, charts)
        return html, charts

    # EML helper kept for completeness
    def package_report_as_eml(self, html_content: str, charts: Dict[str, Path],
                              subject: str, to_email: str, from_email: str) -> str:
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from email.mime.image import MIMEImage

        msg = MIMEMultipart("related")
        msg["Subject"] = subject; msg["To"] = to_email; msg["From"] = from_email
        msg["Date"] = datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S +0000")

        alt = MIMEMultipart("alternative"); msg.attach(alt)
        alt.attach(MIMEText(html_content, "html"))

        for cid, path in charts.items():
            with open(path, "rb") as f:
                img = MIMEImage(f.read())
            img.add_header("Content-ID", f"<{cid}>")
            img.add_header("Content-Disposition", "inline", filename=Path(path).name)
            msg.attach(img)

        eml_path = self.report_output_dir / f"{uuid.uuid4().hex}.eml"
        with open(eml_path, "w") as f:
            f.write(msg.as_string())
        return str(eml_path)


professional_report_generator = ProfessionalReportGenerator()
