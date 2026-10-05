# -*- coding: utf-8 -*-
"""Archive a software reference's README as a PDF in pdf/.

Software cited by its repository has no paper to download, so the README that
documents it is fetched and rendered to PDF; mark_provenance.py can then mark
the passages that support each claim, exactly as for the papers.

Run:  python references/archive_software_readme.py
"""
import urllib.request
from datetime import date
from pathlib import Path

import fitz

REF = Path(__file__).resolve().parent

SOFTWARE = {
    "detoxify2020": "https://raw.githubusercontent.com/unitaryai/detoxify/master/README.md",
}


def main():
    for key, url in SOFTWARE.items():
        text = urllib.request.urlopen(url, timeout=60).read().decode("utf-8")
        header = f"Archived from {url} on {date.today().isoformat()}\n\n"
        doc = fitz.open()
        story = fitz.Story(html="<pre style='font-family:monospace;font-size:8pt;"
                                "white-space:pre-wrap'>"
                                + (header + text).replace("&", "&amp;")
                                .replace("<", "&lt;").replace(">", "&gt;")
                                + "</pre>")
        writer = fitz.DocumentWriter(str(REF / "pdf" / f"{key}.pdf"))
        rect = fitz.paper_rect("a4")
        more = True
        while more:
            dev = writer.begin_page(rect)
            more, _ = story.place(rect + (36, 36, -36, -36))
            story.draw(dev)
            writer.end_page()
        writer.close()
        doc.close()
        print(f"  wrote pdf/{key}.pdf")


if __name__ == "__main__":
    main()
