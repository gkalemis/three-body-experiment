"""Execute exact notebook cells in fresh IPython; no companion modules in cwd."""
import os,json,tempfile
from pathlib import Path
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
root=Path(__file__).resolve().parents[1]
n=nbformat.read(root/'three_body.ipynb',4);nbformat.validate(n)
shell=InteractiveShell.instance()
with tempfile.TemporaryDirectory(prefix='three_body_standalone_') as temp:
    os.chdir(temp)
    for i,cell in enumerate(n.cells):
        if cell.cell_type!='code':continue
        with capture_output() as captured:
            result=shell.run_cell(cell.source,store_history=False)
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(f'Cell {i} failed: {result.error_before_exec or result.error_in_exec}')
        print('Cell',i,'OK',flush=True)
    ui=shell.user_ns['controls']
    assert ui['state']['result'] is not None
    assert ui['state']['result']['report']['numerical_checks_passed']
    assert ui['state']['html'] is not None
    # Colab branch: lightweight compatibility test of the widget-manager setup only.
    import types,sys
    google=types.ModuleType('google');colab=types.ModuleType('google.colab')
    class Output:
        called=False
        def enable_custom_widget_manager(self):self.called=True
    manager=Output();colab.output=manager;google.colab=colab
    sys.modules['google']=google;sys.modules['google.colab']=colab
    with capture_output():shell.run_cell(n.cells[1].source)
    assert manager.called
    # Embedded player exactly equals the shipped standalone template.
    import experiment
    assert experiment.PLAYER_TEMPLATE==(root/'player.html').read_text()
    assert Path('results/interactive/motion.html').exists()
os.chdir(root)
print('STANDALONE IPYTHON PASSED; Colab setup branch tested with a stub, not a live Colab session.')
