from pathlib import Path
import sys, json, time, platform

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, scipy
import experiment as e
from three_body import invariants

out = Path(__file__).resolve().parents[1] / "results/interactive_validation"
out.mkdir(exist_ok=True, parents=True)
records = []
assertions = 0


def check(condition, msg):
    global assertions
    assert condition, msg
    assertions += 1


# Κάθε σετ, και οι δύο διευθύνσεις, 5 μεγέθη: γεωμετρία, αναφορά, ορμή.
for key in e.PRESETS:
    reference = e.initial_pair(key, 0)[0]
    for direction in ["x", "y"]:
        for fraction in [0, 1e-8, 1e-6, 1e-2, 0.1]:
            a, b, meta = e.initial_pair(key, fraction, direction)
            m = e.PRESETS[key]["masses"]
            check(np.array_equal(a, reference), f"A mutated {key}")
            check(np.array_equal(a[6:], b[6:]), "velocities changed")
            check(meta["com_difference"] < 1e-15, "COM shift")
            com = lambda y: np.sum(y[:6].reshape(3, 2) * m[:, None], axis=0) / sum(m)
            check(np.linalg.norm(com(a) - com(b)) < 1e-14, "actual COM changed")
            check(
                np.isclose(meta["max_shift"], fraction * meta["characteristic_length"]),
                "wrong fraction",
            )
            check(
                np.allclose(
                    invariants(a[None], m)[1], invariants(b[None], m)[1], atol=1e-15
                ),
                "momentum changed",
            )
            if fraction == 0:
                check(np.array_equal(a, b), "zero perturbation")

for args in [
    ("bad", 1e-6, "x"),
    ("euler", -0.1, "x"),
    ("euler", float("nan"), "x"),
    ("euler", 0.2, "x"),
    ("euler", 1e-6, "z"),
]:
    try:
        e.initial_pair(*args)
    except ValueError:
        assertions += 1
    else:
        raise AssertionError("invalid input accepted")
for end in [0, -1, float("inf"), float("nan"), 12.01]:
    try:
        e.compare("euler", end=end)
    except ValueError:
        assertions += 1
    else:
        raise AssertionError("invalid duration accepted")

configs = [(key, 1e-6, "x", c["duration"]) for key, c in e.PRESETS.items()]
configs += [
    ("euler_long", 0, "x", 12),
    ("euler_long", 1e-8, "x", 12),
    ("euler_long", 1e-6, "y", 12),
    ("euler_long", 0.01, "x", 12),
    ("euler_long", 0.1, "y", 12),
    ("lagrange_stable", 1e-6, "x", 60),
    ("lagrange_equal", 1e-6, "y", 12),
    ("at_rest", 1e-6, "x", 3),
    ("at_rest", 0, "x", 3),
    ("burrau", 1e-6, "y", 15.8),
    ("burrau", 0.01, "x", 20),
    ("euler", 1e-8, "x", 0.05),
]
for i, (key, fraction, direction, end) in enumerate(configs):
    started = time.time()
    caseout = out / f"case_{i:02}"
    result = e.compare(key, fraction, direction, end, output=caseout)
    r = result["report"]
    r["elapsed_seconds"] = time.time() - started
    check(
        np.isfinite(result["A"]).all() and np.isfinite(result["B"]).all(),
        "nonfinite result",
    )
    check(np.all(np.diff(result["t"]) > 0), "time not monotone")
    check(r["actual_end"] <= end, "extrapolation")
    check(r["com_difference"] < 1e-14, "COM")
    if fraction == 0:
        check(
            np.array_equal(result["A"], result["B"]), "identical initial state diverged"
        )
        check(np.all(result["delta"] == 0), "nonzero D for identical state")
        check(r["final_amplification"] is None, "division by zero")
    if key == "at_rest" and end == 3:
        check(r["stopped_early"], "collision not stopped")
    if key == "euler_long" and fraction == 1e-6 and direction == "x":
        check(r["final_distance"] > 0.5, "no visible divergence")
    if i in [0, 1, 7, 9, 14]:
        e.snapshot(result, caseout / "comparison.png")
        (caseout / "motion.html").write_text(e.animation_html(result, 480))
    if i == 1:
        for frames in [240, 720]:
            (out / f"quality_{frames}.html").write_text(
                e.animation_html(result, frames)
            )
    records.append(r)
    print(
        i,
        key,
        fraction,
        direction,
        end,
        "->",
        r["actual_end"],
        "D",
        r["final_distance"],
        "checks",
        r["numerical_checks_passed"],
        flush=True,
    )
    (out / "validation.json").write_text(
        json.dumps(
            dict(assertions=assertions, records=records), ensure_ascii=False, indent=2
        )
    )

# Reuse actual callbacks as an ordinary test function.
from check_widgets import run_widget_checks

assertions += run_widget_checks()
summary = dict(
    assertions=assertions,
    configuration_count=len(records),
    integrations=sum(3 if r["fraction"] == 0 else 6 for r in records),
    passed_numerical=sum(r["numerical_checks_passed"] for r in records),
    python=platform.python_version(),
    numpy=np.__version__,
    scipy=scipy.__version__,
    records=records,
)
(out / "validation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print("COMPLETE", assertions, "assertions", len(records), "configurations", flush=True)
