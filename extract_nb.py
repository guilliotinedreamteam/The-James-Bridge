import json
with open('neurobridge_py.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
with open('extracted_notebook.md', 'w', encoding='utf-8') as out:
    for cell in nb['cells']:
        if cell['cell_type'] == 'markdown':
            out.write(''.join(cell['source']) + '\n\n')
        elif cell['cell_type'] == 'code':
            out.write('```python\n' + ''.join(cell['source']) + '\n```\n\n')
