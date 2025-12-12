import pypandoc

# Convert markdown to PDF with two-column layout using standard article class
output = pypandoc.convert_file(
    'IEEE_FORMAT_REPORT.md',
    'pdf',
    outputfile='IEEE_FORMAT_REPORT.pdf',
    extra_args=[
        '--pdf-engine=pdflatex',
        '-V', 'documentclass=article',
        '-V', 'classoption=twocolumn',
        '-V', 'geometry:top=0.75in,bottom=1in,left=0.625in,right=0.625in,columnsep=0.25in',
        '-V', 'fontsize=10pt',
        '-V', 'colorlinks=true',
        '-V', 'linkcolor=blue',
        '-V', 'urlcolor=blue'
    ]
)

print("PDF conversion completed successfully!")
print("Output file: IEEE_PUBLICATION_READY_REPORT.pdf")
