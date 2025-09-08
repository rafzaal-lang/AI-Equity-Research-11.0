import os
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

logger = logging.getLogger(__name__)

class ProfessionalReportGenerator:
    """Generate professional, visually appealing equity research reports."""
    
    def __init__(self):
        template_dir = Path(__file__).parent / "templates"
        self.env = Environment(loader=FileSystemLoader(template_dir))
        self.report_output_dir = Path(os.getenv("REPORT_OUTPUT_DIR", "reports"))
        self.report_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style
        sns.set_theme(style="whitegrid")
    
    def _generate_chart(self, data: pd.DataFrame, x: str, y: List[str], title: str, 
                        xlabel: str, ylabel: str, chart_type: str = "line") -> Path:
        """Generate and save a chart."""
        plt.figure(figsize=(10, 6))
        
        if chart_type == "line":
            for col in y:
                sns.lineplot(data=data, x=x, y=col, label=col, marker=".")
        elif chart_type == "bar":
            data.plot(kind="bar", x=x, y=y, rot=45)
        
        plt.title(title, fontsize=16)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.legend()
        plt.tight_layout()
        
        chart_path = self.report_output_dir / f"{uuid.uuid4().hex[:8]}.png"
        plt.savefig(chart_path)
        plt.close()
        
        return chart_path
    
    def _create_charts(self, financial_data: Dict[str, Any]) -> Dict[str, Path]:
        """Create all charts for the report."""
        charts = {}
        
        # Profitability chart
        if "profitability" in financial_data and financial_data.get("periods"):
            df = pd.DataFrame(financial_data["profitability"])
            df["period"] = financial_data["periods"]
            charts["profitability_chart"] = self._generate_chart(
                df, "period", ["gross_margin", "net_margin"], 
                "Profitability Margins", "Period", "Margin (%)"
            )
        
        # Liquidity chart
        if "liquidity" in financial_data and financial_data.get("periods"):
            charts["liquidity_chart"] = self._generate_chart(
                pd.DataFrame({
                    "period": financial_data["periods"],
                    **financial_data["liquidity"]
                }),
                x="period",
                y=list(financial_data["liquidity"].keys()),
                title="Liquidity & Solvency",
                xlabel="Period",
                ylabel="Ratio"
            )

        # DCF sensitivity heatmap (if provided)
        valuation = financial_data.get("valuation") or {}
        sensitivity = valuation.get("sensitivity")  # expect 2D grid (rows x cols)
        if isinstance(sensitivity, dict) and sensitivity.get("z") is not None:
            df = pd.DataFrame(sensitivity["z"],
                              index=sensitivity.get("y_labels") or [],
                              columns=sensitivity.get("x_labels") or [])
            plt.figure()
            sns.heatmap(df, annot=False)
            chart_path = self.report_output_dir / f"{uuid.uuid4().hex[:8]}.png"
            plt.title("DCF Sensitivity")
            plt.tight_layout()
            plt.savefig(chart_path); plt.close()
            charts["dcf_sensitivity_chart"] = chart_path

        # Comps bar chart (EV/EBITDA of top peers if present)
        comps = financial_data.get("comparable_analysis") or financial_data.get("comps") or {}
        peers = comps.get("peers") or []
        if peers:
            df = pd.DataFrame(peers)
            if {"ticker","ev_ebitda"}.issubset(df.columns):
                df = df[["ticker","ev_ebitda"]].dropna().sort_values("ev_ebitda").tail(10)
                plt.figure()
                plt.bar(df["ticker"], df["ev_ebitda"])
                plt.xticks(rotation=45, ha="right"); plt.ylabel("EV/EBITDA")
                plt.title("Comparable Company Multiples")
                plt.tight_layout()
                chart_path = self.report_output_dir / f"{uuid.uuid4().hex[:8]}.png"
                plt.savefig(chart_path); plt.close()
                charts["comps_chart"] = chart_path
        
        return charts
    
    def generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML report from data."""
        try:
            template = self.env.get_template("equity_research_report.html")
            
            # Add generation metadata
            report_data["generation_date"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            report_data["report_id"] = uuid.uuid4().hex
            report_data["report_version"] = "1.0"
            
            html_content = template.render(report_data)
            return html_content
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            raise
    
    def generate_report_with_charts(self, report_data: Dict[str, Any]) -> Tuple[str, Dict[str, Path]]:
        """Generate HTML report and all associated charts."""
        try:
            # Create charts
            charts = self._create_charts(report_data.get("financial_summary", {}))
            
            # Generate HTML
            html_content = self.generate_html_report(report_data)
            
            return html_content, charts
            
        except Exception as e:
            logger.error(f"Error generating report with charts: {e}")
            raise
    
    def package_report_as_eml(self, html_content: str, charts: Dict[str, Path], 
                              subject: str, to_email: str, from_email: str) -> str:
        """Package report as an EML file with embedded charts."""
        msg = MIMEMultipart("related")
        msg["Subject"] = subject
        msg["To"] = to_email
        msg["From"] = from_email
        msg["Date"] = datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S +0000")
        
        # Attach HTML body
        msg_alt = MIMEMultipart("alternative")
        msg.attach(msg_alt)
        
        html_part = MIMEText(html_content, "html")
        msg_alt.attach(html_part)
        
        # Embed charts
        for cid, chart_path in charts.items():
            with open(chart_path, "rb") as f:
                img = MIMEImage(f.read())
                img.add_header("Content-ID", f"<{cid}>")
                img.add_header("Content-Disposition", "inline", filename=chart_path.name)
                msg.attach(img)
        
        eml_path = self.report_output_dir / f"{uuid.uuid4().hex}.eml"
        with open(eml_path, "w") as f:
            f.write(msg.as_string())
        
        return str(eml_path)
    
    def generate_full_report(self, report_data: Dict[str, Any], 
                             output_format: str = "html") -> Path:
        """Generate a full report with charts and save to file."""
        try:
            html_content, charts = self.generate_report_with_charts(report_data)
            
            if output_format == "html":
                report_path = self.report_output_dir / f"{report_data['company']['symbol']}_report.html"
                with open(report_path, "w") as f:
                    f.write(html_content)
                return report_path
            
            elif output_format == "eml":
                subject = f"Equity Research Report: {report_data['company']['name']}"
                eml_path = self.package_report_as_eml(
                    html_content, charts, subject, "<recipient>", "<sender>"
                )
                return Path(eml_path)
            
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
        
        except Exception as e:
            logger.error(f"Error generating full report: {e}")
            raise

# Global instance
professional_report_generator = ProfessionalReportGenerator()

