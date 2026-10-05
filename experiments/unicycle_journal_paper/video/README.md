Video title cards
=================

From the repository root, run:

```bash
python3 experiments/unicycle_journal_paper/video/generate_video_cards.py
```

Requires `pdflatex` with the standard `geometry`, `fontenc`, `graphicx`, `tikz`, and `xcolor`
packages, plus `pdftocairo` from Poppler utilities for PNG export.
Python needs no additional packages.

The script also generates `scenario_1.png`, `scenario_2.png`, and
`scenario_3.png` (with matching PDFs and LaTeX sources). These 1920 × 1080
section cards use white backgrounds, blue text, and the same bold LaTeX font:
two corridors, overlapping circles; multiple corridors; three narrow corridors.

The script creates `intro.pdf`, `outro.pdf`, `intro.png`, and `outro.png` here,
along with their LaTeX sources. PNGs are 1920 × 1080 pixels, ready for video.
Both PDFs are single-page 16:9 cards (320 × 180 mm), using Computer Modern
type with a dark blue background and white text. A rounded white panel below the
authors displays the supplied KU Leuven, MECO, and Flanders Make logos, in that
order, with equally spaced centers. “Real-Robot Validation” uses the title's
bold type style.
Keep the three original logo files beside the script. The outro adds
“Thank you for watching”. The palette is inspired by KU Leuven blue.

Edit the text and color constants in `generate_video_cards.py` to customize
both cards. Use `--output-dir PATH` to save the files elsewhere (the logo files
are also copied there so the generated LaTeX sources can be compiled directly).
