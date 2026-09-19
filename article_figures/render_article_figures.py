"""Reproduce Figures 1-6 and Figures S12-S13 from the accompanying sources."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent


def run(script, *args):
    subprocess.run([sys.executable, str(HERE / script), *map(str, args)], check=True,
                   stdout=subprocess.DEVNULL)


def copy_outputs(source, destination, old, new):
    for ext in ['pdf', 'svg', 'png', 'tif']:
        path = source / f'{old}.{ext}'
        if path.exists():
            shutil.copy2(path, destination / f'{new}.{ext}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='A new directory for the rendered figures.')
    out = parser.parse_args().output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='siteguard_figures_') as temp:
        scratch = Path(temp)
        one, two, legacy, diagnostic = [scratch / name for name in ['fig1', 'fig2', 'legacy', 'diagnostic']]
        run('render_figure1.py', '--output-dir', one)
        run('render_figure2.py', '--output-dir', two)
        run('build_figures.py', '--output-dir', legacy)
        run('build_diagnostic_figure.py', '--source-dir', HERE / 'diagnostic_source',
            '--output-dir', diagnostic)
        copy_outputs(one, out, 'Fig1', 'Fig1')
        copy_outputs(two, out, 'Fig2', 'Fig2')
        for source, article in [('Fig3', 'Fig3'), ('Fig4', 'Fig4'), ('Fig5', 'Fig6'),
                                ('Fig6', 'S12_Fig'), ('Fig7', 'S13_Fig')]:
            copy_outputs(legacy, out, source, article)
        copy_outputs(diagnostic, out, 'Fig5', 'Fig5')
        shutil.copy2(diagnostic / 'Fig5_preview.png', out / 'Fig5.png')
    expected = [f'Fig{i}' for i in range(1, 7)] + ['S12_Fig', 'S13_Fig']
    assert all((out / f'{name}.{ext}').is_file() for name in expected for ext in ['pdf', 'svg'])
    print('Rendered Figures 1-6 and Figures S12-S13.')


if __name__ == '__main__':
    main()
