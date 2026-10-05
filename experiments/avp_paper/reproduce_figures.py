"""Assemble the supplied AVP figures into a single three-panel figure.

Run from any directory:
    python experiments/avp_paper/reproduce_figures.py

The inputs are resolved relative to this script. SVG output embeds the original
SVG drawings and PNG image; it does not redraw or infer the diagram geometry.
PDF and PNG export require Inkscape on PATH (or --inkscape /path/to/inkscape).
Use --formats svg to generate only the SVG without external dependencies.
"""

import argparse
import base64
import os
import shutil
import struct
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

SVG = "http://www.w3.org/2000/svg"
XLINK = "http://www.w3.org/1999/xlink"
FIGURES = Path(__file__).resolve().parent / "figures"
SOURCES = ("mapafinal.svg", "mapbfinal.svg", "map_c_version2.png")
ET.register_namespace("", SVG)
ET.register_namespace("xlink", XLINK)


def element(tag, **attributes):
    return ET.Element(
        "{%s}%s" % (SVG, tag),
        {key.replace("_", "-"): str(value) for key, value in attributes.items()},
    )


def make_figure():
    """Keep equal panel heights and preserve each source's aspect ratio."""
    panel_height, margin, gap, label_height = 600, 16, 24, 38
    panels = []
    for filename in SOURCES:
        path = FIGURES / filename
        if path.suffix == ".svg":
            panel = ET.parse(path).getroot()
            width, height = float(panel.attrib["width"]), float(
                panel.attrib["height"]
            )
            panel.set("viewBox", "0 0 %g %g" % (width, height))
        else:
            data = path.read_bytes()
            width, height = struct.unpack(">II", data[16:24])
            panel = element("image")
            panel.set(
                "{%s}href" % XLINK,
                "data:image/png;base64," + base64.b64encode(data).decode("ascii"),
            )
        panels.append((panel, panel_height * width / height))

    total_width = 2 * margin + sum(width for _, width in panels) + 2 * gap
    total_height = 2 * margin + panel_height + label_height
    figure = element(
        "svg", width=total_width, height=total_height,
        viewBox="0 0 %g %g" % (total_width, total_height),
    )
    title = element("title")
    title.text = "AVP maps: (a) mapafinal, (b) mapbfinal, (c) map_c_version2"
    figure.append(title)
    figure.append(element("rect", width="100%", height="100%", fill="white"))
    x = margin
    for label, (panel, width) in zip(("(a)", "(b)", "(c)"), panels):
        panel.attrib.update(
            x=str(x), y=str(margin), width=str(width), height=str(panel_height)
        )
        figure.append(panel)
        caption = element(
            "text", x=x + width / 2, y=margin + panel_height + 28,
            text_anchor="middle", font_family="serif", font_size=22,
        )
        caption.text = label
        figure.append(caption)
        x += width + gap
    return ET.ElementTree(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=FIGURES / "maps_three_panel",
        help="Output base path, without a file extension.",
    )
    parser.add_argument(
        "--formats", nargs="+", choices=("svg", "pdf", "png"),
        default=("svg", "pdf", "png"),
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution.")
    parser.add_argument("--inkscape", default="inkscape")
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    export_formats = set(args.formats) - {"svg"}
    inkscape = shutil.which(args.inkscape) if export_formats else None
    if export_formats and inkscape is None:
        parser.error("Inkscape is required for PDF/PNG; use --formats svg instead.")

    figure = make_figure()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="avp-figure-") as temporary:
        # Snap terminals and activated environments can inject libraries that
        # are incompatible with the system Inkscape binary. Clean only the
        # child process environment, leaving the user's shell unchanged.
        export_env = dict(os.environ, INKSCAPE_PROFILE_DIR=temporary)
        for variable in (
            "LD_LIBRARY_PATH", "LD_PRELOAD",
            "GTK_PATH", "GTK_MODULES", "GTK_EXE_PREFIX", "GTK_DATA_PREFIX",
            "GTK_IM_MODULE_FILE", "GIO_MODULE_DIR", "GIO_EXTRA_MODULES",
            "GDK_PIXBUF_MODULE_FILE", "GDK_PIXBUF_MODULEDIR",
        ):
            export_env.pop(variable, None)
        svg_path = Path(temporary) / "figure.svg"
        figure.write(svg_path, encoding="utf-8", xml_declaration=True)
        if "svg" in args.formats:
            destination = Path(str(output) + ".svg")
            shutil.copyfile(svg_path, destination)
            print(destination)
        for format_name in sorted(export_formats):
            destination = Path(str(output) + "." + format_name)
            subprocess.run(
                [inkscape, str(svg_path), "--export-area-page",
                 "--export-type=" + format_name,
                 "--export-filename=" + str(destination),
                 "--export-dpi=" + str(args.dpi)],
                env=export_env, check=True,
            )
            print(destination)


if __name__ == "__main__":
    main()
