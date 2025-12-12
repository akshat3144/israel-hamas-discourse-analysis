"""
Convert cleaned IEEE Markdown report to PDF
Uses markdown2 and pdfkit for conversion
"""

import os
import markdown2
import pdfkit

def convert_markdown_to_pdf(md_file, output_pdf):
    """
    Convert markdown file to PDF
    
    Args:
        md_file: Path to markdown file
        output_pdf: Path for output PDF file
    """
    # Read markdown content
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown2.markdown(md_content, extras=['tables', 'fenced-code-blocks', 'metadata'])
    
    # Create full HTML document with styling
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: "Times New Roman", Times, serif;
                font-size: 12pt;
                line-height: 1.6;
                margin: 1in;
                text-align: justify;
            }}
            h1 {{
                font-size: 18pt;
                font-weight: bold;
                margin-top: 0.5in;
                margin-bottom: 0.2in;
            }}
            h2 {{
                font-size: 14pt;
                font-weight: bold;
                margin-top: 0.3in;
                margin-bottom: 0.15in;
            }}
            h3 {{
                font-size: 12pt;
                font-weight: bold;
                margin-top: 0.2in;
                margin-bottom: 0.1in;
            }}
            p {{
                margin-bottom: 0.1in;
                text-indent: 0.2in;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 0.2in 0;
            }}
            th, td {{
                border: 1px solid black;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
                font-weight: bold;
            }}
            img {{
                max-width: 100%;
                height: auto;
                display: block;
                margin: 0.2in auto;
            }}
            code {{
                font-family: "Courier New", monospace;
                background-color: #f5f5f5;
                padding: 2px 4px;
            }}
            pre {{
                background-color: #f5f5f5;
                padding: 10px;
                overflow-x: auto;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Configure pdfkit options for IEEE-style formatting
    options = {
        'page-size': 'Letter',
        'margin-top': '1in',
        'margin-right': '1in',
        'margin-bottom': '1in',
        'margin-left': '1in',
        'encoding': "UTF-8",
        'enable-local-file-access': '',
        'no-outline': None
    }
    
    # Convert HTML to PDF
    try:
        pdfkit.from_string(full_html, output_pdf, options=options)
        print(f"✓ Successfully converted {md_file} to {output_pdf}")
        return True
    except Exception as e:
        print(f"✗ Error converting to PDF: {e}")
        print("\nNote: pdfkit requires wkhtmltopdf to be installed.")
        print("Install it from: https://wkhtmltopdf.org/downloads.html")
        return False

if __name__ == "__main__":
    # Set paths
    md_file = "IEEE_PUBLICATION_READY_REPORT.md"
    output_pdf = "IEEE_PUBLICATION_READY_REPORT.pdf"
    
    # Convert
    print(f"Converting {md_file} to PDF...")
    success = convert_markdown_to_pdf(md_file, output_pdf)
    
    if success:
        print(f"\n✓ PDF generated successfully: {output_pdf}")
    else:
        print("\n✗ PDF conversion failed. Please install required dependencies.")
