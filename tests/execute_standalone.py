"""Execute exact notebook cells in fresh IPython; no companion modules in cwd."""

import os, json, tempfile
from pathlib import Path
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

root = Path(__file__).resolve().parents[1]
n = nbformat.read(root / "three_body.ipynb", 4)
nbformat.validate(n)
shell = InteractiveShell.instance()
with tempfile.TemporaryDirectory(prefix="three_body_standalone_") as temp:
    os.chdir(temp)
    for i, cell in enumerate(n.cells):
        if cell.cell_type != "code":
            continue
        with capture_output() as captured:
            result = shell.run_cell(cell.source, store_history=False)
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(
                f"Cell {i} failed: {result.error_before_exec or result.error_in_exec}"
            )
        print("Cell", i, "OK", flush=True)
    ui = shell.user_ns["controls"]
    assert ui["state"]["result"] is not None
    assert ui["state"]["result"]["report"]["numerical_checks_passed"]
    assert ui["state"]["html"] is not None
    # Το αναγνώσιμο notebook χρησιμοποιεί απευθείας τα ορισμένα κελιά.
    with capture_output():
        ui["choice"].value = "euler_long"
        ui["run"].click()
    long_report = ui["state"]["result"]["report"]
    assert long_report["numerical_checks_passed"]
    assert long_report["final_distance"] > 2
    with capture_output():
        ui["amount"].value = 0
        ui["duration"].value = 0.05
        ui["run"].click()
    assert ui["state"]["result"]["report"]["final_distance"] == 0
    print(
        "Readable callbacks: Euler t=12 D=",
        long_report["final_distance"],
        "; zero change D=0",
        flush=True,
    )
    # Colab branch: lightweight compatibility test of the widget-manager setup only.
    import types, sys

    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")

    class Output:
        called = False

        def enable_custom_widget_manager(self):
            self.called = True

    manager = Output()
    colab.output = manager
    google.colab = colab
    sys.modules["google"] = google
    sys.modules["google.colab"] = colab
    setup = next(
        c.source
        for c in n.cells
        if c.cell_type == "code" and "enable_custom_widget_manager" in c.source
    )
    with capture_output():
        shell.run_cell(setup)
    assert manager.called
    # Embedded player exactly equals the shipped standalone template.
    assert (
        shell.user_ns["PLAYER_TEMPLATE"].lstrip("\n")
        == (root / "player.html").read_text()
    )
    assert Path("results/interactive/motion.html").exists()
os.chdir(root)
print(
    "STANDALONE IPYTHON PASSED; Colab setup branch tested with a stub, not a live Colab session."
)
