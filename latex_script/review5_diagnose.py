# -*- coding: utf-8 -*-
"""
Three final edits, all arising from inspecting the labelling prompts and the
misclassification structure:

  1. Diagnose the Reddit/YouTube accuracy gap - the Reddit prompt carried post
     context, the YouTube prompt carried only an opaque video id.
  2. Replace the vague remedy in Limitations with the specific one.
  3. Add the misclassification-corrected row to the robustness table, and soften
     "near-zero on YouTube" to match the corrected value.

Run:  python latex_script/review5_diagnose.py
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
BODY = HERE / "body_new.tex"

PAIRS = [
    # ---- 1. diagnose the gap ------------------------------------------
    (r"YouTube errors are concentrated in the Neutral class, which the model frequently"
     "\r\n"
     r"reads as Pro-Israel. Short, multilingual, formulaic comments are simply a harder"
     "\r\n"
     r"annotation problem, for the model as for anyone.",

     r"YouTube errors are concentrated in the Neutral class, which the model frequently"
     "\r\n"
     r"reads as Pro-Israel."
     "\r\n"
     "\r\n"
     r"This gap has an identifiable cause in our own pipeline rather than in the"
     "\r\n"
     r"platforms. The Reddit prompt supplied the subreddit, the parent post's title and"
     "\r\n"
     r"its body alongside each comment; the YouTube prompt supplied only an opaque video"
     "\r\n"
     r"identifier, the author handle and a like count. YouTube comments were therefore"
     "\r\n"
     r"annotated without the discursive context that Reddit comments had, and the"
     "\r\n"
     r"instruction to weigh ``video context'' was issued without that context being"
     "\r\n"
     r"present, which plausibly encouraged the model to supply one: a recurring pattern"
     "\r\n"
     r"in the YouTube errors is a justification describing material absent from the"
     "\r\n"
     r"comment. We therefore attribute much of the $0.836$ versus $0.618$ gap to this"
     "\r\n"
     r"asymmetry rather than to an intrinsic difficulty of YouTube text, and note that"
     "\r\n"
     r"our human annotators worked under the same handicap -- they too saw only a video"
     "\r\n"
     r"identifier -- and still agreed among themselves at $\alpha = 0.796$. Short,"
     "\r\n"
     r"multilingual and formulaic comments remain the harder annotation problem, but they"
     "\r\n"
     r"are not the whole of it."),

    # ---- 3a. fourth robustness row ------------------------------------
    (r"    Off-topic removed      & 0.118 & 0.039 & 3.0$\times$ \\",
     r"    Off-topic removed      & 0.118 & 0.039 & 3.0$\times$ \\"
     "\r\n"
     r"    Corrected for error    & 0.207 & 0.072 & 2.9$\times$ \\"),

    # ---- 2. specific remedy in limitations ----------------------------
    (r"Relabelling with a stronger annotator, and a relevance step ahead of",
     r"The clearest routes to tightening these estimates are specific rather than "
     r"general: supplying video-level context to the YouTube annotator, which our "
     r"Reddit prompt had and our YouTube prompt lacked, and inserting an explicit "
     r"relevance decision ahead of"),

    # ---- 3b. soften the overclaim -------------------------------------
    (r"stance and emotional tone is platform-dependent - strong on Reddit, near-zero on",
     r"stance and emotional tone is platform-dependent - strong on Reddit, weak on"),
    (r"negative and only neutrals positive. On YouTube the association is near zero",
     r"negative and only neutrals positive. On YouTube the association is small"),
]

# the third robustness check needs describing in the text too
PROSE = (
    r"Reddit's $V$ falls from $0.149$ to $\textbf{0.082}$ $[0.081, 0.083]$, retaining 55\%",
    r"\textbf{Correcting both platforms for their measured error.} The complementary"
    "\r\n"
    r"analysis disattenuates rather than degrades: treating each platform's measured"
    "\r\n"
    r"confusion matrix as a misclassification process and solving for the underlying"
    "\r\n"
    r"table raises Reddit to $V=0.207$ $[0.169, 0.318]$ and YouTube to $0.072$"
    "\r\n"
    r"$[0.046, 0.171]$, a ratio of $2.9$. Both platforms rise, the contrast persists,"
    "\r\n"
    r"and the corrected YouTube value is small rather than nil -- which is how we"
    "\r\n"
    r"describe it. These intervals are wide because the confusion matrices rest on 73"
    "\r\n"
    r"and 76 items respectively, and at their extremes they touch; we therefore read"
    "\r\n"
    r"this as corroboration of the contrast's direction, not as a precise estimate of"
    "\r\n"
    r"its size."
    "\r\n"
    "\r\n"
    r"Reddit's $V$ falls from $0.149$ to $\textbf{0.082}$ $[0.081, 0.083]$, retaining 55\%",
)


def main():
    with BODY.open("r", encoding="utf-8", newline="") as fh:
        raw = fh.read()
    if "Corrected for error" in raw:
        print("already applied")
        return
    missed = []
    for old, new in [*PAIRS, PROSE]:
        if old in raw:
            raw = raw.replace(old, new, 1)
            print(f"  ok   {old[:58]}")
        else:
            missed.append(old)
            print(f"  MISS {old[:58]}")
    with BODY.open("w", encoding="utf-8", newline="") as fh:
        fh.write(raw)
    print("\nall applied" if not missed else f"\n{len(missed)} missed")


if __name__ == "__main__":
    main()
