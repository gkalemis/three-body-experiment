"""Actual widget callbacks, reusable without source parsing or exec."""

from pathlib import Path
import sys
import types
import IPython.display as ipd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiment as e


def run_widget_checks():
    assertions = 0

    def check(value, message):
        nonlocal assertions
        assert value, message
        assertions += 1

    original = ipd.display
    ipd.display = lambda *a, **k: None
    previous_colab = sys.modules.get("google.colab")
    downloads = []
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(download=lambda path: downloads.append(path))
    sys.modules["google.colab"] = colab
    try:
        ui = e.show_experiment()
        ui["choice"].value = "lagrange_stable"
        check(ui["duration"].max == 60, "preset duration cap")
        ui["duration"].value = 31.25
        check(ui["endfield"].value == 31.25, "slider->number")
        ui["endfield"].value = 30.5
        check(ui["duration"].value == 30.5, "number->slider")
        ui["choice"].value = "euler_long"
        check(ui["duration"].value == 12 and ui["duration"].max == 12, "preset reset")
        ui["amount"].value = -1
        check(not ui["epsilon"].disabled, "custom slider disabled")
        ui["epsilon"].value = 0.01
        ui["direction"].value = "y"
        ui["reset"].click()
        check(
            ui["amount"].value == 1e-6 and ui["direction"].value == "x", "reset failed"
        )
        ui["run"].click()
        check(ui["state"]["result"] is not None, "callback execution failed")
        check(
            ui["state"]["result"]["report"]["numerical_checks_passed"],
            "callback numerical checks",
        )
        check(ui["state"]["html"] is not None, "automatic animation failed")
        for quality in [240, 720, 480]:
            ui["quality"].value = quality
            ui["movie_button"].click()
            check(ui["state"]["html"] is not None, "animation quality callback")
        ui["amount"].value = 0
        check(
            ui["state"]["result"] is None and ui["movie_button"].disabled,
            "stale result retained",
        )
        ui["duration"].value = 0.05
        ui["run"].click()
        check(ui["state"]["result"]["report"]["final_distance"] == 0, "zero callback")
        # Τεχνητή αποτυχία μόνο για έλεγχο αποκατάστασης των χειριστηρίων.
        real = e.compare

        def fail(*a, **k):
            raise RuntimeError("controlled test failure")

        e.compare = fail
        ui["run"].click()
        e.compare = real
        check(
            ui["state"]["result"] is None
            and not ui["run"].disabled
            and not ui["choice"].disabled,
            "failure recovery",
        )
        ui["run"].click()
        check(ui["state"]["result"] is not None, "recovery next run failed")
        ui["download"].click()
        check(len(downloads) == 1, "Colab download dispatch")
        ui["quality"].value = 240
        ui["download"].click()
        check(len(downloads) == 2, "rebuild before download")
        ui["direction"].value = "y"
        check(ui["download"].disabled, "stale download enabled")
    finally:
        ipd.display = original
        if previous_colab is None:
            sys.modules.pop("google.colab", None)
        else:
            sys.modules["google.colab"] = previous_colab
    return assertions


if __name__ == "__main__":
    print("WIDGETS PASSED", run_widget_checks())
