import subprocess
import os
import pypdf

test_html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
@page { size: A4; margin: 20mm; }
body { font-family: sans-serif; font-size: 14pt; }
</style>
</head>
<body>
<h1>Test PDF Generation</h1>
<p>Testing headless browser PDF export.</p>
</body>
</html>"""

with open("scratch/test.html", "w", encoding="utf-8") as f:
    f.write(test_html)

chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
if not os.path.exists(chrome_path):
    chrome_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

pdf_out = os.path.abspath("scratch/test.pdf")
if os.path.exists(pdf_out):
    os.remove(pdf_out)

cmd = [
    chrome_path,
    "--headless",
    "--disable-gpu",
    "--no-pdf-header-footer",
    f"--print-to-pdf={pdf_out}",
    os.path.abspath("scratch/test.html")
]

res = subprocess.run(cmd, capture_output=True, text=True)
print("Return code:", res.returncode)
print("PDF exists:", os.path.exists(pdf_out))
if os.path.exists(pdf_out):
    r = pypdf.PdfReader(pdf_out)
    print("Pages:", len(r.pages))
