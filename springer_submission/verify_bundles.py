# -*- coding: utf-8 -*-
"""Check both archives byte-for-byte against the files on disk, and confirm the
compiled PDF is newer than every source that feeds it.

Run this before uploading anywhere.
"""
import hashlib
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def md5(b):
    return hashlib.md5(b).hexdigest()


ok = True
for zname in ("jcss_submission.zip", "overleaf_project.zip"):
    zp = HERE / zname
    if not zp.exists():
        print(f"{zname}: MISSING")
        ok = False
        continue
    z = zipfile.ZipFile(zp)
    stale = []
    for n in z.namelist():
        disk = HERE / n
        if not disk.exists() or md5(z.read(n)) != md5(disk.read_bytes()):
            stale.append(n)
    size = zp.stat().st_size / 1e6
    print(f"{zname:24} {len(z.namelist()):>3} files  {size:5.2f} MB  "
          f"{'IN SYNC' if not stale else 'STALE: ' + ', '.join(stale)}")
    if stale:
        ok = False

# the PDF must be at least as new as everything that produces it
pdf = HERE / "sn_paper.pdf"
newer = [p.name for p in (HERE / "sn_paper.tex", HERE / "refs.bib",
                          HERE / "sn-jnl.cls", HERE / "sn-basic.bst")
         if p.exists() and p.stat().st_mtime > pdf.stat().st_mtime + 1]
print(f"\nsn_paper.pdf {'is current' if not newer else 'is OLDER than: ' + ', '.join(newer)}")
if newer:
    ok = False

log = (HERE / "sn_paper.log").read_text(encoding="utf-8", errors="ignore")
pages = [l for l in log.splitlines() if "Output written" in l]
print(f"  {pages[0].strip() if pages else 'no output line in log'}")
print(f"  overfull boxes      : {log.count('Overfull')}")
print(f"  undefined citations : {log.lower().count('citation') and sum('undefined' in l.lower() and 'citation' in l.lower() for l in log.splitlines())}")
print(f"  undefined references: {sum('undefined' in l.lower() and 'reference' in l.lower() for l in log.splitlines())}")

print("\n" + ("READY TO UPLOAD" if ok else "NOT READY - rebuild or repackage first"))
sys.exit(0 if ok else 1)
