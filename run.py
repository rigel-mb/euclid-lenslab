"""Download once, then execute the two notebooks and build the overview."""
from pathlib import Path
import argparse
import os
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))


def execute_notebook(path):
    """Execute cells with IPython, retaining real outputs without a kernel server."""
    import nbformat
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output

    notebook = nbformat.read(path, as_version=4)
    shell = InteractiveShell.instance()
    shell.reset(new_session=False)
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        count += 1
        with capture_output(display=True) as captured:
            result = shell.run_cell(cell.source, store_history=True)
        error = result.error_before_exec or result.error_in_exec
        if error:
            raise RuntimeError(f'{path.name}, cell {count}: {error}') from error
        cell.execution_count = count
        cell.outputs = []
        if captured.stdout:
            cell.outputs.append(nbformat.v4.new_output('stream', name='stdout', text=captured.stdout))
        if captured.stderr:
            cell.outputs.append(nbformat.v4.new_output('stream', name='stderr', text=captured.stderr))
        for displayed in captured.outputs:
            cell.outputs.append(nbformat.v4.new_output('display_data', data=displayed.data,
                                                       metadata=displayed.metadata))
    nbformat.validate(notebook)
    nbformat.write(notebook, path)


def main():
    os.chdir(ROOT)
    for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        os.environ.setdefault(variable, '2')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='Acquire public data into the local cache, then stop.')
    args = parser.parse_args()
    if args.download:
        from lenslab.data import acquire
        from lenslab.encoder import acquire as acquire_representations
        from lenslab.expanded_sampling import acquire as acquire_sample
        acquire(ROOT, images=True)
        acquire_representations(ROOT)
        acquire_sample(ROOT)
        return
    for path in sorted((ROOT / 'notebooks').glob('*.ipynb')):
        print(f'Executing {path.name}', flush=True)
        execute_notebook(path)
    from lenslab.report import build
    print(f'Ready: {build(ROOT)}')


if __name__ == '__main__':
    main()
