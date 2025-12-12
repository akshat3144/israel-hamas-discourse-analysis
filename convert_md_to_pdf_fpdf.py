"""
Convert cleaned IEEE Markdown report to PDF using FPDF2
Pure Python solution without external dependencies
"""

import os
import re
from fpdf import FPDF

class IEEE_PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=25.4)  # 1 inch = 25.4mm
        
    def header(self):
        pass  # No header for IEEE format
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Times', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def clean_markdown_formatting(text):
    """Remove markdown formatting for plain text rendering"""
    # Remove image/figure markdown
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # Remove links but keep text
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    # Remove bold/italic
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'\1', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'___(.+?)___', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    text = re.sub(r'_(.+?)_', r'\1', text)
    # Remove inline code
    text = re.sub(r'`(.+?)`', r'\1', text)
    return text

def convert_markdown_to_pdf(md_file, output_pdf):
    """
    Convert markdown file to PDF using FPDF2
    
    Args:
        md_file: Path to markdown file
        output_pdf: Path for output PDF file
    """
    try:
        # Read markdown content
        with open(md_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Create PDF
        pdf = IEEE_PDF()
        pdf.add_page()
        pdf.set_font('Times', '', 12)
        
        in_code_block = False
        skip_next = False
        
        for i, line in enumerate(lines):
            if skip_next:
                skip_next = False
                continue
                
            line = line.rstrip()
            
            # Handle code blocks
            if line.startswith('```'):
                in_code_block = not in_code_block
                continue
            
            if in_code_block:
                pdf.set_font('Courier', '', 9)
                pdf.multi_cell(0, 5, line)
                pdf.set_font('Times', '', 12)
                continue
            
            # Skip YAML front matter
            if line.startswith('---') and i < 20:
                continue
            
            # Skip image lines
            if line.startswith('![') or (line.startswith('*Fig.') and not line.startswith('**')):
                continue
            
            # Handle headings
            if line.startswith('# '):
                pdf.ln(5)
                pdf.set_font('Times', 'B', 16)
                text = clean_markdown_formatting(line[2:])
                pdf.multi_cell(0, 8, text, align='C')
                pdf.ln(3)
                pdf.set_font('Times', '', 12)
                
            elif line.startswith('## '):
                pdf.ln(4)
                pdf.set_font('Times', 'B', 14)
                text = clean_markdown_formatting(line[3:])
                pdf.multi_cell(0, 7, text)
                pdf.ln(2)
                pdf.set_font('Times', '', 12)
                
            elif line.startswith('### '):
                pdf.ln(3)
                pdf.set_font('Times', 'BI', 12)
                text = clean_markdown_formatting(line[4:])
                pdf.multi_cell(0, 6, text)
                pdf.ln(2)
                pdf.set_font('Times', '', 12)
                
            # Handle horizontal rules
            elif line.startswith('---') or line.startswith('***'):
                pdf.ln(2)
                pdf.line(25, pdf.get_y(), 185, pdf.get_y())
                pdf.ln(2)
                
            # Handle tables (simplified)
            elif '|' in line and not line.startswith(' '):
                pdf.set_font('Times', '', 10)
                text = clean_markdown_formatting(line)
                pdf.multi_cell(0, 5, text)
                pdf.set_font('Times', '', 12)
                
            # Handle lists
            elif line.strip().startswith('- ') or line.strip().startswith('* '):
                text = clean_markdown_formatting(line.strip()[2:])
                current_x = pdf.get_x()
                pdf.set_x(30)
                pdf.multi_cell(0, 5, f"• {text}")
                pdf.set_x(current_x)
                
            # Handle numbered lists
            elif re.match(r'^\d+\.', line.strip()):
                text = clean_markdown_formatting(line.strip())
                current_x = pdf.get_x()
                pdf.set_x(30)
                pdf.multi_cell(0, 5, text)
                pdf.set_x(current_x)
                
            # Handle math equations (simplified - just show placeholder)
            elif line.startswith('$$'):
                pdf.set_font('Times', 'I', 11)
                pdf.multi_cell(0, 5, '[Mathematical equation]')
                pdf.set_font('Times', '', 12)
                
            # Handle blockquotes
            elif line.startswith('>'):
                pdf.set_font('Times', 'I', 11)
                text = clean_markdown_formatting(line[1:].strip())
                pdf.multi_cell(0, 5, text, border='L')
                pdf.set_font('Times', '', 12)
                
            # Handle regular paragraphs
            elif line.strip():
                text = clean_markdown_formatting(line)
                # Check if it's bold text (like **RQ1**)
                if line.startswith('**') and '**' in line[2:]:
                    pdf.set_font('Times', 'B', 12)
                pdf.multi_cell(0, 5, text)
                pdf.set_font('Times', '', 12)
                
            # Handle empty lines
            else:
                pdf.ln(3)
        
        # Save PDF
        pdf.output(output_pdf)
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
    print(f"Converting {md_file} to PDF using FPDF2...")
    print("Note: Images will not be included in this PDF version.")
    print("For a complete version with images, please use a Markdown to PDF converter with image support.")
    print()
    success = convert_markdown_to_pdf(md_file, output_pdf)
    
    if success:
        abs_path = os.path.abspath(output_pdf)
        print(f"\n✓ PDF generated successfully!")
        print(f"✓ File saved at: {abs_path}")
    else:
        print("\n✗ PDF conversion failed.")
