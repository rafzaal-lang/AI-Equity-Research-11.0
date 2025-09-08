# src/services/report/professional_report_generator.py
import os
import re
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    import seaborn as sns
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False

from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)

_REQUIRED_CIDS = [
    "profitability_chart",
    "liquidity_chart",
    "dcf_sensitivity_chart",
    "comps_chart",
]

class ProfessionalReportGenerator:
    """
    Generates HTML (or EML-ready HTML) for your Jinja template.
    Ensures the four CID images referenced by the template always exist.
    """

    def __init__(self):
        template_dir = Path(__file__).parent / "templates"
        self.env = Environment(loader=FileSystemLoader(template_dir), autoescape=True)
        self.output_dir = Path(os.getenv("REPORT_OUTPUT_DIR", "reports"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if _HAS_SNS:
            sns.set_theme(style="whitegrid")

    # ---------------- util: figures ----------------
    def _new_fig(self, w=10, h=6):
        plt.figure(figsize=(w, h))

    def _save_chart(self) -> Path:
        p = self.output_dir / f"{uuid.uuid4().hex[:8]}.png"
        plt.tight_layout()
        plt.savefig(p, bbox_inches="tight", dpi=150)
        plt.close()
        return p

    def _placeholder(self, title: str) -> Path:
        self._new_fig()
        plt.axis("off")
        plt.text(0.5, 0.5, f"{title}\nNot available", ha="center", va="center", fontsize=14)
        return self._save_chart()

    # ---------------- chart builders ----------------
    def _line(self, df: pd.DataFrame, x: str, y_cols: List[str], title: str, xlabel: str, ylabel: str) -> Path:
        self._new_fig()
        for col in y_cols:
            if _HAS_SNS:
                sns.lineplot(data=df, x=x, y=col, label=col, marker=".")
            else:
                plt.plot(df[x], df[col], label=col, marker=".")
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

    def _create_charts(self, fin: Dict[str, Any]) -> Dict[str, Path]:
        charts: Dict[str, Path] = {}

        # Profitability
        try:
            periods = fin.get("periods") or []
            prof = fin.get("profitability") or {}
            if periods and isinstance(prof, dict) and any(isinstance(v, list) for v in prof.values()):
                df = pd.DataFrame({"period": periods, **prof})
                ycols = [c for c in df.columns if c != "period"]
                if ycols:
                    charts["profitability_chart"] = self._line(df, "period", ycols, "Profitability", "Period", "Value")
        except Exception:
            pass

        # Liquidity
        try:
            periods = fin.get("periods") or []
            liq = fin.get("liquidity") or {}
            if periods and isinstance(liq, dict) and any(isinstance(v, list) for v in liq.values()):
                df = pd.DataFrame({"period": periods, **liq})
                ycols = [c for c in df.columns if c != "period"]
                if ycols:
                    charts["liquidity_chart"] = self._line(df, "period", ycols, "Liquidity & Solvency", "Period", "Ratio")
        except Exception:
            pass

        # DCF Sensitivity
        try:
            sens = (fin.get("valuation") or {}).get("sensitivity") or {}
            if sens.get("z") and sens.get("x_labels") and sens.get("y_labels"):
                charts["dcf_sensitivity_chart"] = self._heatmap(
                    sens["z"], sens["x_labels"], sens["y_labels"], "DCF EV Sensitivity"
                )
        except Exception:
            pass

        # Comps
        try:
            comps = (fin.get("comparable_analysis") or {}).get("peers") or []
            if comps:
                df = pd.DataFrame(comps)
                metric = None; label = None
                if "ev_ebitda" in df.columns and df["ev_ebitda"].notna().any():
                    metric = "ev_ebitda"; label = "EV/EBITDA"
                elif "pe" in df.columns and df["pe"].notna().any():
                    metric = "pe"; label = "P/E"
                if metric:
                    df = df[["symbol", metric]].dropna().sort_values(metric).tail(12)
                    self._new_fig()
                    plt.bar(df["symbol"], df[metric])
                    plt.xticks(rotation=45, ha="right"); plt.ylabel(label); plt.title("Comparable Company Multiples")
                    charts["comps_chart"] = self._save_chart()
        except Exception:
            pass

        # Ensure all required CIDs exist with placeholders
        for cid in _REQUIRED_CIDS:
            if cid not in charts:
                charts[cid] = self._placeholder(cid.replace("_", " ").title())

        return charts

    # ---------------- renderers ----------------
    def generate_html(self, template_name: str, report_data: Dict[str, Any],
                      charts: Dict[str, Path], inline_for_web: bool = True) -> str:
        """
        Render your Jinja template. If `inline_for_web=True`, we replace
        'cid:...' srcs with file paths so it shows in a browser. For EML packaging,
        pass inline_for_web=False and then use `package_report_as_eml(...)`.
        """
        template = self.env.get_template(template_name)
        html = template.render(report_data)

        if inline_for_web:
            # Replace cid:foobar with actual file paths
            for cid, path in charts.items():
                html = html.replace(f"cid:{cid}", str(path))
        return html

    def generate_all(self, report_data: Dict[str, Any], inline_for_web: bool = True) -> Tuple[str, Dict[str, Path]]:
        fin = report_data.get("financial_summary") or {}
        charts = self._create_charts(fin)
        html = self.generate_html("equity_research_report.html", report_data, charts, inline_for_web=inline_for_web)
        return html, charts

    def package_report_as_eml(self, html_content: str, charts: Dict[str, Path],
                              subject: str, to_email: str, from_email: str) -> str:
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from email.mime.image import MIMEImage

        msg = MIMEMultipart("related")
        msg["Subject"] = subject
        msg["To"] = to_email
        msg["From"] = from_email
        msg["Date"] = datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S +0000")

        alt = MIMEMultipart("alternative")
        msg.attach(alt)
        alt.attach(MIMEText(html_content, "html"))

        # Attach the images with CIDs matching template
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


professional_report_generator = ProfessionalReportGenerator()
