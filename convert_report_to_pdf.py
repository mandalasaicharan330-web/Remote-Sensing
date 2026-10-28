"""
convert_report_to_pdf.py
========================
Directly converts `report/final_project_report.md` into a professionally
formatted, publication-ready PDF: `report/final_project_report.pdf` (and `report/Final_Report.pdf`).
"""

import os
import sys
import re
import subprocess
import pypdf
import markdown

if sys.stdout:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def convert_md_to_pdf():
    md_path = "report/final_project_report.md"
    if not os.path.exists(md_path):
        print(f"Error: {md_path} not found.")
        return False

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Pre-process math and special notations for clean HTML rendering
    # Double dollar blocks
    def math_block_replace(match):
        code = match.group(1).strip()
        # Clean common LaTeX syntax for standard display
        clean = (code.replace(r"\mathcal{L}_{\text{BCE}}", "L<sub>BCE</sub>")
                     .replace(r"\mathbf{y}", "<b>y</b>")
                     .replace(r"\mathbf{z}", "<b>z</b>")
                     .replace(r"\sigma", "&sigma;")
                     .replace(r"\sum_{k=1}^K", "&sum;<sub>k=1..K</sub>")
                     .replace(r"\left[", "[")
                     .replace(r"\right]", "]")
                     .replace(r"\frac{1}{K}", "(1/K)")
                     .replace(r"\in", "&isin;")
                     .replace(r"\dots", "...")
                     .replace(r"\top", "<sup>T</sup>")
                     .replace(r"\hat{p}", "p&#770;")
                     .replace(r"\hat{y}", "y&#770;")
                     .replace(r"\ge", "&ge;")
                     .replace(r"\mathbb{I}", "&Iopf;"))
        return f'<div class="math-eq">{clean}</div>'

    md_text = re.sub(r'\$\$(.*?)\$\$', math_block_replace, md_text, flags=re.DOTALL)

    # Inline math
    def math_inline_replace(match):
        code = match.group(1)
        clean = (code.replace(r"M_1", "<i>M</i><sub>1</sub>")
                     .replace(r"M_2", "<i>M</i><sub>2</sub>")
                     .replace(r"M_3", "<i>M</i><sub>3</sub>")
                     .replace(r"B02", "B02")
                     .replace(r"B03", "B03")
                     .replace(r"B04", "B04")
                     .replace(r"B08", "B08")
                     .replace(r"B11", "B11")
                     .replace(r"B12", "B12")
                     .replace(r"\mathbf{y}", "<b>y</b>")
                     .replace(r"\mathbf{X}", "<b>X</b>")
                     .replace(r"\mathbf{z}", "<b>z</b>")
                     .replace(r"T=0.50", "<i>T</i> = 0.50")
                     .replace(r"T=0.5", "<i>T</i> = 0.50")
                     .replace(r"T^*", "<i>T</i><sup>*</sup>")
                     .replace(r"T_k", "<i>T<sub>k</sub></i>")
                     .replace(r"\times", "&times;")
                     .replace(r"\text{m}", " m")
                     .replace(r"\in", "&isin;")
                     .replace(r"\dots", "...")
                     .replace(r"\mathbb{R}", "&Ropf;")
                     .replace(r"\top", "<sup>T</sup>")
                     .replace(r"\hat{p}", "p&#770;")
                     .replace(r"\hat{y}", "y&#770;")
                     .replace(r"\ge", "&ge;")
                     .replace(r"\sigma", "&sigma;"))
        return f'<span class="inline-math">{clean}</span>'

    md_text = re.sub(r'\$(.*?)\$', math_inline_replace, md_text)

    # Convert to HTML via Python-Markdown
    body_html = markdown.markdown(
        md_text,
        extensions=['tables', 'fenced_code', 'nl2br']
    )

    # Wrap in HTML5 template with academic styling and page layout
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Technical Report: Multi-Label Land-Cover Recognition Using Sentinel-2 Multispectral Classification</title>
<style>
@page {{
    size: A4 portrait;
    margin: 18mm 18mm 18mm 18mm;
    @bottom-right {{
        content: counter(page);
    }}
}}
* {{
    box-sizing: border-box;
}}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 9.5pt;
    line-height: 1.45;
    color: #1e293b;
    margin: 0;
    padding: 0;
}}
h1 {{
    font-size: 18pt;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.25;
    margin-top: 0;
    margin-bottom: 8pt;
    border-bottom: 2pt solid #1e40af;
    padding-bottom: 6pt;
}}
h2 {{
    font-size: 13pt;
    font-weight: 700;
    color: #1e40af;
    border-bottom: 1.5pt solid #cbd5e1;
    padding-bottom: 3pt;
    margin-top: 14pt;
    margin-bottom: 7pt;
    break-after: avoid;
}}
h3 {{
    font-size: 11pt;
    font-weight: 700;
    color: #0f766e;
    margin-top: 10pt;
    margin-bottom: 4pt;
    break-after: avoid;
}}
h4 {{
    font-size: 10pt;
    font-weight: 700;
    color: #334155;
    margin-top: 8pt;
    margin-bottom: 3pt;
    break-after: avoid;
}}
p {{
    margin-top: 0;
    margin-bottom: 7pt;
    text-align: justify;
}}
ul, ol {{
    margin-top: 0;
    margin-bottom: 7pt;
    padding-left: 18pt;
}}
li {{
    margin-bottom: 3.5pt;
}}
hr {{
    border: none;
    border-top: 1pt solid #e2e8f0;
    margin: 12pt 0;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 6pt;
    margin-bottom: 9pt;
    font-size: 8.5pt;
}}
th, td {{
    border: 0.75pt solid #cbd5e1;
    padding: 4.5pt 6pt;
    text-align: left;
}}
th {{
    background-color: #1e40af;
    color: #ffffff;
    font-weight: 600;
}}
tr:nth-child(even) td {{
    background-color: #f8fafc;
}}
code {{
    font-family: "Consolas", "Courier New", monospace;
    font-size: 8.5pt;
    background-color: #f1f5f9;
    padding: 1pt 3pt;
    border-radius: 2pt;
    color: #0f766e;
}}
pre {{
    background-color: #f8fafc;
    border: 0.75pt solid #e2e8f0;
    border-radius: 4pt;
    padding: 6pt 9pt;
    font-size: 8pt;
    line-height: 1.35;
    margin-top: 4pt;
    margin-bottom: 8pt;
    font-family: "Consolas", monospace;
}}
.math-eq {{
    background: #f8fafc;
    border: 0.5pt solid #e2e8f0;
    padding: 5pt 10pt;
    border-radius: 3pt;
    text-align: center;
    font-family: "Cambria Math", "Times New Roman", serif;
    font-size: 10pt;
    margin: 7pt 0;
}}
.inline-math {{
    font-family: "Cambria Math", "Times New Roman", serif;
    font-size: 9.5pt;
}}
blockquote {{
    border-left: 3pt solid #1e40af;
    padding-left: 8pt;
    margin: 6pt 0;
    color: #475569;
    background-color: #f8fafc;
    padding-top: 4pt;
    padding-bottom: 4pt;
}}
/* Page break rules for clean layout */
h2#section-2, h2:has(+ h3) {{
    break-before: auto;
}}
</style>
</head>
<body>
{body_html}
</body>
</html>
"""

    temp_html = "scratch/temp_markdown_report.html"
    with open(temp_html, "w", encoding="utf-8") as f:
        f.write(html_doc)

    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    if not os.path.exists(chrome_path):
        chrome_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

    pdf_out = os.path.abspath("report/final_project_report.pdf")
    pdf_legacy = os.path.abspath("report/Final_Report.pdf")

    cmd = [
        chrome_path,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_out}",
        os.path.abspath(temp_html)
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error converting to PDF: {res.stderr}")
        return False

    if os.path.exists(pdf_out):
        reader = pypdf.PdfReader(pdf_out)
        page_count = len(reader.pages)
        size_kb = os.path.getsize(pdf_out) / 1024.0

        print("=" * 60)
        print("[OK] PDF Conversion Successful!")
        print(f"  Target File: {pdf_out}")
        print(f"  Page Count : {page_count} pages")
        print(f"  File Size  : {size_kb:.1f} KB")
        print("=" * 60)
        return True
        return True

    return False

if __name__ == "__main__":
    convert_md_to_pdf()
