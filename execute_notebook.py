"""Проверяет и выполняет весь ноутбук. Не включает массовый inference автоматически."""
from pathlib import Path
import asyncio
import sys
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parent
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
path = root/'experiment.ipynb'
nb = nbformat.read(path, as_version=4)
nbformat.validate(nb)
client = NotebookClient(nb, timeout=300, kernel_name='python3', resources={'metadata': {'path': str(root)}})
try:
    client.execute()
except Exception:
    nbformat.write(nb, root/'results/notebook_failed.ipynb')
    raise
nbformat.write(nb, path)
errors = [o for c in nb.cells if c.cell_type == 'code' for o in c.get('outputs', []) if o.output_type == 'error']
if errors:
    raise RuntimeError(f'Notebook errors: {len(errors)}')
print('Executed all notebook cells successfully.')
