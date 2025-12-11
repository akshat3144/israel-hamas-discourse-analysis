"""
Convert cleaned IEEE Markdown report to PDF using WeasyPrint
Alternative approach that doesn't require external dependencies
"""

import os
import markdown
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration

def convert_markdown_to_pdf(md_file, output_pdf):
    """
    Convert markdown file to PDF using WeasyPrint
    
    Args:
        md_file: Path to markdown file
        output_pdf: Path for output PDF file
    """
    try:
        # Read markdown content
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Convert markdown to HTML with extensions
        html_content = markdown.markdown(
            md_content, 
            extensions=['extra', 'codehilite', 'tables', 'toc']
        )
        
        # Create full HTML document with IEEE-style CSS
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                @page {{
                    size: letter;
                    margin: 1in;
                }}
                body {{
                    font-family: "Times New Roman", Times, serif;
                    font-size: 10pt;
                    line-height: 1.5;
                    text-align: justify;
                    color: #000;
                }}
                h1 {{
                    font-size: 16pt;
                    font-weight: bold;
                    margin-top: 18pt;
                    margin-bottom: 12pt;
                    text-align: center;
                }}
                h2 {{
                    font-size: 12pt;
                    font-weight: bold;
                    margin-top: 14pt;
                    margin-bottom: 10pt;
                }}
                h3 {{
                    font-size: 11pt;
                    font-weight: bold;
                    margin-top: 12pt;
                    margin-bottom: 8pt;
                    font-style: italic;
                }}
                p {{
                    margin-bottom: 8pt;
                    text-indent: 0.2in;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 12pt 0;
                    font-size: 9pt;
                }}
                th, td {{
                    border: 1px solid #000;
                    padding: 6pt;
                    text-align: left;
                }}
                th {{
                    background-color: #e0e0e0;
                    font-weight: bold;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    display: block;
                    margin: 12pt auto;
                }}
                code {{
                    font-family: "Courier New", Courier, monospace;
                    font-size: 9pt;
                    background-color: #f5f5f5;
                    padding: 2pt 4pt;
                }}
                pre {{
                    font-family: "Courier New", Courier, monospace;
                    font-size: 9pt;
                    background-color: #f5f5f5;
                    padding: 10pt;
                    overflow-x: auto;
                    margin: 8pt 0;
                }}
                hr {{
                    border: none;
                    border-top: 1px solid #000;
                    margin: 12pt 0;
                }}
                blockquote {{
                    margin: 8pt 20pt;
                    padding-left: 10pt;
                    border-left: 3pt solid #ccc;
                }}
                em {{
                    font-style: italic;
                }}
                strong {{
                    font-weight: bold;
                }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        # Initialize font configuration
        font_config = FontConfiguration()
        
        # Convert HTML to PDF
        HTML(string=full_html, base_url=os.path.dirname(os.path.abspath(md_file))).write_pdf(
            output_pdf,
            font_config=font_config
        )
        
        print(f"✓ Successfully converted {md_file} to {output_pdf}")
        return True
        
    except Exception as e:
        print(f"✗ Error converting to PDF: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Set paths
    md_file = "IEEE_PUBLICATION_READY_REPORT.md"
    output_pdf = "IEEE_PUBLICATION_READY_REPORT.pdf"
    
    # Convert
    print(f"Converting {md_file} to PDF using WeasyPrint...")
    success = convert_markdown_to_pdf(md_file, output_pdf)
    
    if success:
        print(f"\n✓ PDF generated successfully: {output_pdf}")
        print(f"✓ File saved at: {os.path.abspath(output_pdf)}")
    else:
        print("\n✗ PDF conversion failed.")
