"""Combine the thesis width and circle-shift figures, preserving PDF vectors.

The panels have equal heights, with minimum widths first. Larger annotations
are generated specifically for this composition; standalone exports stay intact.
"""

from pathlib import Path
import shutil
import subprocess
import tempfile

import matplotlib.pyplot as plt
import min_width
import max_moving


def main():
    directory = Path(__file__).resolve().parent
    name = 'min_width_and_max_moving'
    with tempfile.TemporaryDirectory(prefix='kappa-thesis-panels-') as temporary:
        for module, source in ((min_width, 'min_width_recreated.pdf'),
                               (max_moving, 'max_moving_recreated.pdf')):
            figure = module.create_figure(font_scale=1.20)
            with plt.rc_context(module.STYLE):
                figure.savefig(Path(temporary) / source,
                               bbox_inches='tight', pad_inches=0.04)
            plt.close(figure)
        shutil.copyfile(directory / f'{name}.tex', Path(temporary) / f'{name}.tex')
        result = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
             f'-output-directory={temporary}', f'{name}.tex'],
            cwd=temporary, capture_output=True, text=True,
        )
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        output = directory / f'{name}.pdf'
        shutil.copyfile(Path(temporary) / f'{name}.pdf', output)
    subprocess.run(
        ['pdftoppm', '-singlefile', '-png', '-r', '150',
         str(output), str(directory / name)], check=True,
    )
    print(f'Saved {output}')
    print(f'Saved {directory / (name + ".png")}')


if __name__ == '__main__':
    main()
