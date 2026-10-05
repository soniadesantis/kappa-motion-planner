#!/usr/bin/env python3
"""Generate PDF and 1920 × 1080 PNG video cards with pdflatex and pdftocairo."""

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile


# Approximate KU Leuven-inspired palette; these are not official branded cards.
BACKGROUND = "006A93"
FOREGROUND = "FFFFFF"
ACCENT = "71C5E8"
LOGOS = (
    "KULEUVEN_LOGO_2012.pdf",
    "MECO-logo-no-background.png",
    "Flanders make logo.jpg",
)

TEMPLATE = r"""\documentclass{article}
\usepackage[paperwidth=320mm,paperheight=180mm,margin=18mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{xcolor}
\usepackage{graphicx}
\usepackage{tikz}
\definecolor{background}{HTML}{@BACKGROUND@}
\definecolor{foreground}{HTML}{@FOREGROUND@}
\definecolor{accent}{HTML}{@ACCENT@}
\pagecolor{background}
\color{foreground}
\pagestyle{empty}
\setlength{\parindent}{0pt}
\begin{document}
\centering
\vspace*{0mm}
{\fontsize{16}{20}\selectfont\bfseries Real-Robot Validation\par}
\vspace{6mm}
{\color{accent}\rule{32mm}{0.8mm}\par}
\vspace{7mm}
{\fontsize{28}{34}\selectfont\bfseries
A Fast Analytical Trajectory Planner\\
for Unicycle Robots Through Safe Rectangular\\
Corridors Based on Time-Optimal Motion Primitives\par}
\vspace{8mm}
{\fontsize{16}{22}\selectfont
Sonia De Santis, Alejandro Astudillo, Alejandro Gonzalez-Garcia,\\
Wilm Decr\'{e}, Jan Swevers\par}
\vspace{12mm}
% Equal center-to-center spacing, with MECO at the center of the panel.
\begin{tikzpicture}[x=1mm,y=1mm]
\fill[white,rounded corners=4mm] (-113,-15.5) rectangle (113,15.5);
\node[inner sep=0] at (-74,0)
  {\includegraphics[width=54mm]{KULEUVEN_LOGO_2012.pdf}};
\node[inner sep=0] at (0,0)
  {\includegraphics[width=72mm]{MECO-logo-no-background.png}};
\node[inner sep=0] at (74,0)
  {\includegraphics[width=64mm,trim=100bp 130bp 100bp 130bp,clip]{Flanders make logo.jpg}};
\end{tikzpicture}\par
\vspace{-5mm}
\vfill
@CLOSING@
\vspace*{4mm}
\end{document}
"""

SCENARIO_TEMPLATE = r"""\documentclass{article}
\usepackage[paperwidth=320mm,paperheight=180mm,margin=18mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{xcolor}
\definecolor{foreground}{HTML}{@BLUE@}
\definecolor{accent}{HTML}{@ACCENT@}
\pagecolor{white}
\color{foreground}
\pagestyle{empty}
\setlength{\parindent}{0pt}
\begin{document}
\centering
\vspace*{\fill}
{\fontsize{44}{52}\selectfont\bfseries Scenario @NUMBER@\par}
\vspace{8mm}
{\color{accent}\rule{32mm}{0.8mm}\par}
\vspace{10mm}
{\fontsize{40}{48}\selectfont\bfseries @DESCRIPTION@\par}
\vspace*{\fill}
\end{document}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent,
        help="Output folder (default: the folder containing this script).",
    )
    args = parser.parse_args()
    compiler = shutil.which("pdflatex")
    if compiler is None:
        parser.error("pdflatex is required. Install a LaTeX distribution first.")
    renderer = shutil.which("pdftocairo")
    if renderer is None:
        parser.error("pdftocairo is required. Install Poppler utilities first.")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    asset_dir = Path(__file__).resolve().parent
    for logo in LOGOS:
        if not (asset_dir / logo).is_file():
            parser.error(f"Missing logo: {asset_dir / logo}")
        if output_dir != asset_dir:
            shutil.copyfile(asset_dir / logo, output_dir / logo)

    cards = []
    for name, closing in (
        ("intro", ""),
        ("outro", r"{\fontsize{19}{24}\selectfont\itshape Thank you for watching\par}"),
    ):
        source = TEMPLATE
        for key, value in {
            "BACKGROUND": BACKGROUND, "FOREGROUND": FOREGROUND,
            "ACCENT": ACCENT, "CLOSING": closing,
        }.items():
            source = source.replace(f"@{key}@", value)
        cards.append((name, source))
    for number, description in enumerate((
        r"Two Corridors, Overlapping Circles",
        r"Multiple Corridors",
        r"Three Narrow Corridors",
    ), start=1):
        source = SCENARIO_TEMPLATE
        for key, value in {
            "BLUE": BACKGROUND, "ACCENT": ACCENT,
            "NUMBER": str(number), "DESCRIPTION": description,
        }.items():
            source = source.replace(f"@{key}@", value)
        cards.append((f"scenario_{number}", source))

    for name, source in cards:
        # Keep generated TeX for easy manual adjustment; build debris stays temporary.
        tex_path = output_dir / f"{name}.tex"
        tex_path.write_text(source, encoding="utf-8")
        with tempfile.TemporaryDirectory(prefix="video-cards-") as build_dir:
            for logo in LOGOS:
                shutil.copyfile(asset_dir / logo, Path(build_dir) / logo)
            result = subprocess.run(
                [compiler, "-interaction=nonstopmode", "-halt-on-error",
                 "-no-shell-escape", f"-output-directory={build_dir}", str(tex_path)],
                cwd=build_dir, capture_output=True, text=True,
            )
            if result.returncode:
                raise SystemExit(f"LaTeX failed for {name}:\n{result.stdout}\n{result.stderr}")
            if "Overfull" in result.stdout:
                raise SystemExit(f"Content overflow in {name}:\n{result.stdout}")
            pdf_path = output_dir / f"{name}.pdf"
            shutil.copyfile(Path(build_dir) / f"{name}.pdf", pdf_path)
            print(f"Generated {pdf_path}")
            subprocess.run(
                [renderer, "-png", "-singlefile", "-scale-to-x", "1920",
                 "-scale-to-y", "1080", str(pdf_path), str(output_dir / name)],
                check=True,
            )
            print(f"Generated {output_dir / f'{name}.png'}")


if __name__ == "__main__":
    main()
