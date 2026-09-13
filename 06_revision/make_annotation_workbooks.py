# -*- coding: utf-8 -*-
"""
Build one Excel workbook per annotator for the 270-item validation sample.

Each workbook has a Guidelines sheet (the rulebook plus a worked example) and an
Annotate sheet holding the blind items, with a P/I/N/R dropdown on the label
column and a live progress counter. Nothing reveals the model's label.

Run:  python 06_revision/make_annotation_workbooks.py
Out:  06_revision/validation/annotator_A.xlsx, annotator_B.xlsx, annotator_C.xlsx
"""
import math
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

HERE = Path(__file__).resolve().parent
VAL = HERE / "validation"
BLIND = VAL / "validation_blind.csv"
ANNOTATORS = ("A", "B", "C")

FONT = "Arial"
HDR_ROW = 4
FIRST = HDR_ROW + 1

INK = Font(name=FONT, size=10)
INK_B = Font(name=FONT, size=10, bold=True)
TITLE = Font(name=FONT, size=13, bold=True)
MUTED = Font(name=FONT, size=9, color="666666")

HDR_FILL = PatternFill("solid", fgColor="E8E4DC")
TODO_FILL = PatternFill("solid", fgColor="FFF6D6")   # unlabelled rows
INPUT_FILL = PatternFill("solid", fgColor="FFFFCC")  # cells you fill in

THIN = Side(style="thin", color="D0CCC4")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

COLS = [
    ("item_id", 10),
    ("platform", 11),
    ("context", 34),
    ("comment text", 78),
    ("human_label", 13),
    ("notes", 28),
]

LABEL_DEFS = [
    ("P", "Supports Palestine",
     "Advocates the rights, interests or perspectives of Palestinians - statehood, "
     "sovereignty, self-determination - or criticises Israeli conduct from that position.",
     "“Hamas wanted to negotiate the week of the 7th. USA/Israel said no.”"),
    ("I", "Supports Israel",
     "Supportive of Israel's interests, security, sovereignty or right to defend itself, "
     "or criticises Hamas and Palestinian actors from that position.",
     "“Israel doesnt occupy Gaza, for the past 18 years. They pulled settlers and "
     "military out in 2005.”"),
    ("N", "Neutral / unclear",
     "On topic, but takes no definite side - even-handed, questioning, or too ambiguous "
     "to place on either side.",
     "“Not a progressive and IDF stalled negotiations and brought back the fighting.”"),
    ("R", "Irrelevant",
     "Not about the conflict at all - spam, off-topic chatter, unrelated personal remarks.",
     "“What's your favorite pizza topping?”"),
]

RULES = [
    "Filter out R first, then decide between P / I / N.",
    "Judge the comment as written, in the context shown. Do not infer a stance the text "
    "does not actually express.",
    "Some comments are fragments of a longer exchange. Label what is there; if that is "
    "genuinely not enough to place a side, it is N.",
    "Work independently. Do not discuss items with the other annotators while labelling - "
    "independent agreement is precisely what is being measured.",
    "Do not open validation_key.csv. It contains the model's labels and would bias you.",
    "Leave a note on any item you found genuinely hard. Those notes are useful in the paper.",
]


def guidelines_sheet(wb, who):
    ws = wb.create_sheet("Guidelines", 0)
    ws.sheet_view.showGridLines = False
    for col, w in zip("ABCD", (6, 22, 74, 52)):
        ws.column_dimensions[col].width = w

    ws["A1"] = "Stance annotation - guidelines"
    ws["A1"].font = TITLE
    ws["A2"] = f"Annotator {who}  |  270 items  |  Israel-Hamas discourse validation sample"
    ws["A2"].font = MUTED

    ws["A4"] = "What to do"
    ws["A4"].font = INK_B
    steps = [
        "Go to the Annotate sheet.",
        "For each row, read the comment in column D and choose a label in column E "
        "(yellow). Column E has a dropdown - P, I, N or R.",
        "Optionally add a note in column F.",
        "Rows you have not labelled stay highlighted, so you can always see what is left. "
        "The counter at the top of the sheet tracks progress.",
        "When all 270 are done, save the file (keep the .xlsx name) and hand it back.",
    ]
    r = 5
    for i, s in enumerate(steps, 1):
        ws.cell(r, 1, f"{i}.").font = INK
        c = ws.cell(r, 2, s)
        c.font = INK
        c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.merge_cells(start_row=r, start_column=2, end_row=r, end_column=4)
        ws.row_dimensions[r].height = 28
        r += 1

    r += 1
    ws.cell(r, 1, "Labels").font = INK_B
    r += 1
    for j, h in enumerate(("", "Meaning", "When it applies", "Example"), 1):
        c = ws.cell(r, j, h)
        c.font = INK_B
        c.fill = HDR_FILL
        c.border = BOX
    r += 1
    for code, meaning, desc, example in LABEL_DEFS:
        for j, v in enumerate((code, meaning, desc, example), 1):
            c = ws.cell(r, j, v)
            c.font = INK_B if j == 1 else INK
            c.alignment = Alignment(wrap_text=True, vertical="top")
            c.border = BOX
        ws.row_dimensions[r].height = 58
        r += 1

    r += 1
    ws.cell(r, 1, "Rules").font = INK_B
    r += 1
    for s in RULES:
        ws.cell(r, 1, "•").font = INK
        c = ws.cell(r, 2, s)
        c.font = INK
        c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.merge_cells(start_row=r, start_column=2, end_row=r, end_column=4)
        ws.row_dimensions[r].height = 26
        r += 1

    r += 1
    ws.cell(r, 1, "Example of a filled row").font = INK_B
    r += 1
    ex = [
        ("item_id", "V0001"),
        ("platform", "reddit"),
        ("context", "r/AskMiddleEast"),
        ("comment text", "hopefully bibi will come to his senses and stop this madness."),
        ("human_label", "P"),
        ("notes", "criticises Netanyahu's conduct of the war"),
    ]
    for field, value in ex:
        ws.cell(r, 2, field).font = MUTED
        c = ws.cell(r, 3, value)
        c.font = INK_B if field == "human_label" else INK
        if field == "human_label":
            c.fill = INPUT_FILL
        c.alignment = Alignment(wrap_text=True, vertical="top")
        r += 1
    ws.cell(r + 1, 2, "Only the yellow cells (human_label, and optionally notes) "
                      "are yours to fill in.").font = MUTED
    return ws


def annotate_sheet(wb, who, df):
    ws = wb.create_sheet("Annotate", 1)
    n = len(df)
    last = FIRST + n - 1

    for i, (name, width) in enumerate(COLS, 1):
        ws.column_dimensions[get_column_letter(i)].width = width

    ws["A1"] = f"Annotate - annotator {who}"
    ws["A1"].font = TITLE
    ws["A2"] = ("Choose P / I / N / R in the yellow human_label column. "
                "Unlabelled rows stay highlighted.")
    ws["A2"].font = MUTED

    lab_col = get_column_letter(5)
    rng = f"${lab_col}${FIRST}:${lab_col}${last}"
    ws["E1"] = "labelled"
    ws["E1"].font = MUTED
    ws["F1"] = (f'=COUNTIF({rng},"P")+COUNTIF({rng},"I")'
                f'+COUNTIF({rng},"N")+COUNTIF({rng},"R")')
    ws["F1"].font = Font(name=FONT, size=13, bold=True)
    ws["E2"] = "remaining"
    ws["E2"].font = MUTED
    ws["F2"] = f"=COUNTA($A${FIRST}:$A${last})-$F$1"
    ws["F2"].font = INK_B

    for i, (name, _) in enumerate(COLS, 1):
        c = ws.cell(HDR_ROW, i, name)
        c.font = INK_B
        c.fill = HDR_FILL
        c.border = BOX
        c.alignment = Alignment(vertical="center")
    ws.row_dimensions[HDR_ROW].height = 20

    top_wrap = Alignment(wrap_text=True, vertical="top")
    for k, row in enumerate(df.itertuples(index=False), start=FIRST):
        ctx = row.context_subreddit or ""
        if ctx:
            ctx = f"r/{ctx}"
            if row.context_post_title:
                ctx += f"\n“{row.context_post_title}”"
        elif row.context_video_id:
            ctx = f"video {row.context_video_id}"

        text = str(row.text)
        if text[:1] in "=+@":          # never let a comment parse as a formula
            text = "'" + text

        values = [row.item_id, row.platform, ctx, text, None, None]
        for i, v in enumerate(values, 1):
            c = ws.cell(k, i, v)
            c.font = INK
            c.alignment = top_wrap
            c.border = BOX
            if i in (5, 6):
                c.fill = INPUT_FILL
        ws.cell(k, 5).alignment = Alignment(horizontal="center", vertical="center")
        lines = max(math.ceil(len(text) / 95), ctx.count("\n") + 1)
        ws.row_dimensions[k].height = 15 * min(max(lines, 2), 9)

    dv = DataValidation(
        type="list", formula1='"P,I,N,R"', allow_blank=True, showDropDown=False,
        errorTitle="Not a valid label",
        error="Use P (Palestine), I (Israel), N (neutral/unclear) or R (irrelevant).",
        promptTitle="Stance", prompt="P / I / N / R")
    dv.error_style = "stop"
    ws.add_data_validation(dv)
    dv.add(f"{lab_col}{FIRST}:{lab_col}{last}")

    ws.conditional_formatting.add(
        f"A{FIRST}:F{last}",
        FormulaRule(formula=[f"AND($A{FIRST}<>\"\",$E{FIRST}=\"\")"],
                    fill=TODO_FILL, stopIfTrue=False))

    ws.freeze_panes = f"A{FIRST}"
    ws.auto_filter.ref = f"A{HDR_ROW}:F{last}"
    return ws


def main():
    df = pd.read_csv(BLIND).fillna("")
    print(f"{len(df)} items from {BLIND.name}")

    for who in ANNOTATORS:
        wb = Workbook()
        wb.remove(wb.active)
        guidelines_sheet(wb, who)
        annotate_sheet(wb, who, df)
        wb.active = 1
        out = VAL / f"annotator_{who}.xlsx"
        wb.save(out)
        print(f"  wrote {out.name}  ({out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
