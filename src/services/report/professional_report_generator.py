# src/services/report/professional_report_generator.py
from __future__ import annotations

import os
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# Use a headless backend BEFORE importing pyplot (needed on servers like Render/Heroku)
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# Seaborn is optional; fall back gracefully if missing
try:
    import seaborn as sns
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False

logger = logging.getLogger(__name__)

# The template references these 4 Content-IDs; we always produce an image for each.
_REQUIRED_CIDS = [
    "profitability_chart",
    "liquidity_chart",
    "dcf_sensitivity_chart",
    "comps_chart",
]


class ProfessionalReportGenerator:
    """
    Generate professional HTML (web) or EML-ready HTML using the Jinja template.
    - Ensures charts exist for the 4 required CIDs (uses placeholders when data is missing).
    - New API:   generate_all(report_data, inline_for_web=True) -> (html, charts)
    - Legacy API: generate_report_with_charts(report_data) and generate_html_report(report_data)
    """

    def __init__(self) -> None:
        template_dir = Path(__file__).parent / "templates"
        self.env = Environment(loader=FileSystemLoader(template_dir), autoescape=True)

        self.output_dir = Path(os.getenv("REPORT_OUTPUT_DIR", "reports"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if _HAS_SNS:
            sns.set_theme(style="whitegrid")

    # ---------------- utilities: figures/files ----------------
    def _new_fig(self, w: float = 10, h: float = 6) -> None:
        plt.figure(figsize=(w, h))

    def _save_chart(self) -> Path:
        out = self.output_dir / f"{uuid.uuid4().hex[:8]}.png"
        plt.tight_layout()
        plt.savefig(out, bbox_inches="tight", dpi=150)
        plt.close()
        return out

    def _placeholder(self, title: str) -> Path:
        self._new_fig()
        plt.axis("off")
        plt.text(
            0.5, 0.5,
            f"{title}\nNot available",
            ha="center", va="center", fontsize=14
        )
        return self._save_chart()

    # ---------------- chart builders ----------------
    def _line(
        self,
        df: pd.DataFrame,
        x: str,
        y_cols: List[str],
        title: str,
        xlabel: str,
        ylabel: str,
    ) -> Path:
        self._new_fig()
        for col in y_cols:
            if col not in df.columns:
                continue
            if _HAS_SNS:
                sns.lineplot(data=df, x=x, y=col, label=col, marker=".")
            else:
                plt.plot(df[x], df[col], label=col, marker=".")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        return self._save_chart()

    def _heatmap(
        self,
        z: List[List[Optional[float]]],
        x_labels: List[str],
        y_labels: List[str],
        title: str,
    ) -> Path:
        self._new_fig()
        # Replace None with nan for plotting
        z_df = pd.DataFrame(z, index=y_labels, columns=x_labels, dtype="float64")
        if _HAS_SNS:
            sns.heatmap(z_df, annot=False)
        else:
            plt.imshow(z_df.values, aspect="auto", origin="upper")
            plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha="right")
            plt.yticks(range(len(y_labels)), y_labels)
            plt.colorbar()
        plt.title(title)
        return self._save_chart()

    # ---------------- chart orchestration ----------------
    def _create_charts(self, fin: Dict[str, Any]) -> Dict[str, Path]:
        charts: Dict[str, Path] = {}

        # Profitability chart (expects arrays keyed by metric + periods list)
        try:
            periods = fin.get("periods") or []
            prof = fin.get("profitability") or {}
            if periods and isinstance(prof, dict) and any(isinstance(v, list) for v in prof.values()):
                df = pd.DataFrame({"period": periods, **prof})
                ycols = [c for c in df.columns if c != "period"]
                if ycols:
                    charts["profitability_chart"] = self._line(
                        df, "period", ycols, "Profitability Margins", "Period", "Value"
                    )
        except Exception as e:
            logger.debug("Profitability chart skipped: %s", e)

        # Liquidity chart
        try:
            periods = fin.get("periods") or []
            liq = fin.get("liquidity") or {}
            if periods and isinstance(liq, dict) and any(isinstance(v, list) for v in liq.values()):
                df = pd.DataFrame({"period": periods, **liq})
                ycols = [c for c in df.columns if c != "period"]
                if ycols:
                    charts["liquidity_chart"] = self._line(
                        df, "period", ycols, "Liquidity & Solvency", "Period", "Ratio"
                    )
        except Exception as e:
            logger.debug("Liquidity chart skipped: %s", e)

        # DCF sensitivity heatmap (expects {"z": 2D, "x_labels": [...], "y_labels": [...]})
        try:
            sens = (fin.get("valuation") or {}).get("sensitivity") or {}
            if sens.get("z") and sens.get("x_labels") and sens.get("y_labels"):
                charts["dcf_sensitivity_chart"] = self._heatmap(
                    sens["z"], sens["x_labels"], sens["y_labels"], "DCF EV Sensitivity"
                )
        except Exception as e:
            logger.debug("Sensitivity chart skipped: %s", e)

        # Comparable multiples (prefer EV/EBITDA else P/E)
        try:
            comps = (fin.get("comparable_analysis") or {}).get("peers") or []
            if comps:
                df = pd.DataFrame(comps)
                metric, label = None, None
                if "ev_ebitda" in df.columns and df["ev_ebitda"].notna().any():
                    metric, label = "ev_ebitda", "EV/EBITDA"
                elif "pe" in df.columns and df["pe"].notna().any():
                    metric, label = "pe", "P/E"

                if metric:
                    df = df[["symbol", metric]].dropna().sort_values(metric).tail(12)
                    self._new_fig()
                    plt.bar(df["symbol"], df[metric])
                    plt.xticks(rotation=45, ha="right")
                    plt.ylabel(label)
                    plt.title("Comparable Company Multiples")
                    charts["comps_chart"] = self._save_chart()
        except Exception as e:
            logger.debug("Comps chart skipped: %s", e)

        # Ensure all required CIDs exist
        for cid in _REQUIRED_CIDS:
            if cid not in charts:
                charts[cid] = self._placeholder(cid.replace("_", " ").title())

        return charts

    # ---------------- renderers ----------------
    def generate_html(
        self,
        template_name: str,
        report_data: Dict[str, Any],
        charts: Dict[str, Path],
        inline_for_web: bool = True,
    ) -> str:
        """
        Render the Jinja template. If inline_for_web=True, replace cid: URLs with
        file paths for browser display. For EML packaging, keep cid: URLs intact.
        """
        template = self.env.get_template(template_name)
        html = template.render(report_data)

        if inline_for_web:
            for cid, path in charts.items():
                html = html.replace(f"cid:{cid}", str(path))
        return html

    def generate_all(
        self,
        report_data: Dict[str, Any],
        inline_for_web: bool = True,
    ) -> Tuple[str, Dict[str, Path]]:
        fin = report_data.get("financial_summary") or {}
        charts = self._create_charts(fin)
        html = self.generate_html(
            "equity_research_report.html",
            report_data,
            charts,
            inline_for_web=inline_for_web,
        )
        return html, charts

    # -------- Legacy compatibility (used by older UI fallbacks) --------
    def generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """Legacy: render template without inlining cid images (keeps cid: links)."""
        template = self.env.get_template("equity_research_report.html")
        return template.render(report_data)

    def generate_report_with_charts(
        self, report_data: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Path]]:
        """Legacy: make charts + return (html_with_cids, charts)."""
        fin = report_data.get("financial_summary") or {}
        charts = self._create_charts(fin)
        html = self.generate_html_report(report_data)  # keep cid: references
        return html, charts

    # ---------------- EML packaging ----------------
    def package_report_as_eml(
        self,
        html_content: str,
        charts: Dict[str, Path],
        subject: str,
        to_email: str,
        from_email: str,
    ) -> str:
        msg = MIMEMultipart("related")
        msg["Subject"] = subject
        msg["To"] = to_email
        msg["From"] = from_email
        msg["Date"] = datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S +0000")

        alt = MIMEMultipart("alternative")
        msg.attach(alt)
        alt.attach(MIMEText(html_content, "html"))

        # Attach the images so cid:profitability_chart, etc. resolve in mail clients
        for cid in _REQUIRED_CIDS:
            path = charts.get(cid)
            if not path:
                continue
            with open(path, "rb") as f:
                img = MIMEImage(f.read())
            img.add_header("Content-ID", f"<{cid}>")
            img.add_header("Content-Disposition", "inline", filename=Path(path).name)
            msg.attach(img)

        eml_path = self.output_dir / f"{uuid.uuid4().hex}.eml"
        with open(eml_path, "w") as f:
            f.write(msg.as_string())
        return str(eml_path)


# Global instance used by the UI
professional_report_generator = ProfessionalReportGenerator()
