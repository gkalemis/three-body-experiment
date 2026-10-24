"""Execute the standalone student edition and real widget callbacks."""
from pathlib import Path
import os
import sys
import tempfile
import types
import json
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

ROOT = Path(__file__).resolve().parents[1]
notebook = nbformat.read(ROOT / 'three_body_students.ipynb', 4)
nbformat.validate(notebook)
shell = InteractiveShell.instance()
count = 0

def check(value, message):
    global count
    assert value, message
    count += 1

with tempfile.TemporaryDirectory(prefix='student_notebook_') as folder:
    os.chdir(folder)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != 'code':
            continue
        check(cell.metadata['jupyter']['source_hidden'], 'source not hidden')
        check(cell.metadata['cellView'] == 'form', 'Colab form metadata missing')
        with capture_output():
            result = shell.run_cell(cell.source)
        check(not result.error_before_exec and not result.error_in_exec, f'cell {index} failed')
    ui = shell.user_ns['student_controls']
    r = ui['state']['result']['report']
    check(r['numerical_checks_passed'] and r['final_distance'] > .8, 'default example')
    check(ui['extras'].selected_index is None, 'details expanded by default')
    check(ui['state']['html'] is not None, 'animation absent')
    check('Θέλω να δω και το διάγραμμα' in ui['state']['html'], 'optional chart missing')
    check('| Διαφορά D =' not in ui['state']['html'], 'technical clock visible')
    with capture_output():
        ui['choice'].value = 'euler_long'
        ui['run'].click()
    check(ui['state']['result']['report']['final_distance'] > 2, 'long visible difference')
    before = ui['state']['result']
    with capture_output():
        ui['quality'].value = 240
    check(ui['state']['result'] is before, 'quality reintegrated')
    with capture_output():
        ui['change'].value = 0
    check(ui['state']['result'] is None and ui['download'].disabled, 'stale result')
    with capture_output():
        ui['endfield'].value = .05
        ui['run'].click()
    check(ui['duration'].value == .05, 'time fields not linked')
    check(ui['state']['result']['report']['final_distance'] == 0, 'A=B')
    with capture_output():
        ui['choice'].value = 'at_rest'
        ui['duration'].value = 3
        ui['run'].click()
    check(ui['state']['result']['report']['stopped_early'], 'stop absent')
    check('Σταματήσαμε' in ui['message'].value, 'early stop hidden')
    with capture_output():
        ui['reset'].click()
    check(ui['duration'].value == 1.4 and ui['state']['result'] is None, 'reset')
    # Explicitly test the visible warning, rather than silently accepting bad checks.
    real = shell.user_ns['compare']
    def flagged(*args, **kwargs):
        output = real(*args, **kwargs)
        output['report']['numerical_checks_passed'] = False
        return output
    shell.user_ns['compare'] = flagged
    with capture_output():
        ui['duration'].value = .05
        ui['run'].click()
    check('διερευνητική' in ui['message'].value, 'failed checks not visible')
    shell.user_ns['compare'] = real
    page = ui['state']['html']
    (ROOT / 'results/interactive_validation/student_motion.html').write_text(page)
    # A standalone module view and notebook definitions share the same algorithm.
    check(Path('results/students/motion.html').exists(), 'HTML not saved')
    os.chdir(ROOT)
(ROOT / 'results/interactive_validation/student_checks.json').write_text(json.dumps({
    'assertions': count, 'standalone_ipython': True,
    'live_colab_tested': False, 'live_jupyterlab_frontend_tested': False,
}, indent=2))
print('STUDENT EDITION PASSED', count)
