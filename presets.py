"""Επτά αρχικές διατάξεις και συμβατοί αριθμητικοί έλεγχοι.

Τοπική αριθμητική εκτέλεση: python presets.py euler.
Το διαδραστικό μενού και ο μοναδικός player ορίζονται στο experiment.py.
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from three_body import center, rhs, integrate, distance, diagnostics, PAIRS


def make_presets():
    """Κάθε πακέτο δίνει μάζες, θέσεις, ταχύτητες και διάρκεια."""
    triangle = np.array(
        [[0.0, 2 / np.sqrt(3)], [-1.0, -1 / np.sqrt(3)], [1.0, -1 / np.sqrt(3)]]
    )  # Πλευρά 2.

    def rotating(label, masses, positions, omega, duration, explanation):
        m = np.array(masses, dtype=float)
        r = center(positions, m)
        v = omega * np.column_stack((-r[:, 1], r[:, 0]))
        return dict(
            label=label,
            masses=m,
            r=r,
            v=v,
            omega=omega,
            duration=duration,
            explanation=explanation,
        )

    equal = rotating(
        "Ισόπλευρο τρίγωνο — ίσες μάζες",
        [1, 1, 1],
        triangle,
        np.sqrt(3 / 8),
        10.0,
        "Το τρίγωνο περιστρέφεται με σταθερό σχήμα και μέγεθος. "
        "Είναι σχετική ισορροπία: φαίνεται ακίνητο αν περιστρεφόμαστε μαζί του. "
        "Για ίσες μάζες είναι ασταθές· μια μικρή διαταραχή μπορεί να ενισχυθεί. "
        "Αυτό δεν σημαίνει ότι θα φανεί μεγάλη αλλαγή σε τόσο μικρή διάρκεια.",
    )
    unequal = rotating(
        "Ισόπλευρο τρίγωνο — ένα βαρύ σώμα",
        [0.001, 0.01, 1],
        triangle,
        np.sqrt(1.011 / 8),
        30.0,
        "Και εδώ το τρίγωνο διατηρεί σχήμα και μέγεθος καθώς περιστρέφεται. "
        "Η συγκεκριμένη αναλογία μαζών ικανοποιεί το κλασικό κριτήριο "
        "γραμμικής ευστάθειας στο επίπεδο. Αυτό αφορά απειροστά μικρές "
        "διαταραχές και δεν εγγυάται κάθε κίνηση για πάντα.",
    )
    line = rotating(
        "Ευθεία Euler — περιστρεφόμενη διάταξη",
        [1, 1, 1],
        np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]),
        np.sqrt(1.25),
        6.0,
        "Το μεσαίο σώμα μένει στο κέντρο μάζας και τα άλλα δύο "
        "περιστρέφονται διατηρώντας την ευθεία. Είναι ασταθής σχετική "
        "ισορροπία. Οι βαρυτικές δυνάμεις στα εξωτερικά σώματα δεν μηδενίζονται: "
        "δίνουν την επιτάχυνση της κυκλικής κίνησης.",
    )
    rest = dict(
        label="Τρίγωνο αρχικά ακίνητο — αρχίζει να συρρικνώνεται",
        masses=np.ones(3),
        r=triangle.copy(),
        v=np.zeros((3, 2)),
        omega=None,
        duration=1.4,
        explanation="Ίδια γεωμετρία, αλλά χωρίς αρχική περιστροφή. "
        "Τα σώματα ξεκινούν ακίνητα και έλκονται προς το κέντρο: "
        "δεν βρίσκονται σε στατική ισορροπία. Παρακολουθούμε μόνο "
        "το διάστημα πριν από τη σύγκρουση. Δεν συνεχίζουμε μέσα από αυτήν.",
    )
    burrau = dict(
        label="Burrau — ενίσχυση μικρής αρχικής διαφοράς",
        masses=np.array([3.0, 4.0, 5.0]),
        r=np.array([[1.0, 3.0], [-2.0, -1.0], [1.0, -1.0]]),
        v=np.zeros((3, 2)),
        omega=None,
        duration=15.8,
        explanation="Το αρχικό πείραμα του notebook. Διαδοχικές στενές "
        "προσεγγίσεις ενισχύουν τη μικρή αρχική διαφορά στο επιλεγμένο "
        "χρονικό διάστημα. Η απόκλιση μόνη της δεν αποδεικνύει χάος.",
    )
    visible = dict(line)
    visible.update(
        label="Ξεκίνα εδώ — εμφανής απόκλιση Euler (t=8)",
        duration=8.0,
        animation_interval_ms=250,
        explanation="Ίδιες μάζες, αρχικές θέσεις, ταχύτητες και διαταραχή 10⁻⁶ "
        "με την περίπτωση Euler t=6, αλλά παρακολουθούμε μέχρι t=8. "
        "Η μεγαλύτερη διάρκεια αφήνει την αστάθεια να γίνει ορατή. "
        "Στην Α η ευθεία περιστρέφεται, ενώ η Β απομακρύνεται εμφανώς "
        "από αυτή τη διάταξη. Παρατήρησε ιδιαίτερα το διάστημα t=6 έως t=8. "
        "Η αρχική αναπαραγωγή διαρκεί περίπου 30 δευτερόλεπτα. "
        "Η απόκλιση από ασταθή σχετική ισορροπία δεν αποδεικνύει από μόνη της χάος.",
    )
    longer = dict(line)
    longer.update(
        label="Euler για περισσότερο χρόνο (t=12)",
        duration=12.0,
        animation_interval_ms=250,
        animation_frames=240,
        explanation="Ξεκινάμε όπως στην Euler t=6 και t=8, με την ίδια μικρή "
        "αλλαγή θέσης 10⁻⁶. Αφήνουμε όμως τα σώματα να κινηθούν μέχρι t=12. "
        "Η Α παραμένει σε περιστρεφόμενη ευθεία, ενώ η Β ακολουθεί πια "
        "καθαρά διαφορετική διαδρομή. Οι εικόνες Α και Β βρίσκονται "
        "σε δύο χωριστά πλαίσια· τα αχνά ίχνη και στα δύο είναι συνεχή. "
        "Ο περισσότερος χρόνος κίνησης μεγαλώνει τη διαφορά. Το πιο αργό "
        "πάτημα αναπαραγωγής αλλάζει μόνο πόσο γρήγορα βλέπουμε την εικόνα. "
        "Η διαφορά δείχνει αστάθεια και από μόνη της δεν αποδεικνύει χάος.",
    )
    return dict(
        euler_visible=visible,
        euler_long=longer,
        lagrange_stable=unequal,
        lagrange_equal=equal,
        euler=line,
        at_rest=rest,
        burrau=burrau,
    )


PRESETS = make_presets()


def preset_initial(case, perturbed=False, epsilon=1e-6):
    m = case["masses"]
    r = case["r"].copy()
    if perturbed:
        r[0, 0] += epsilon
        r[1, 0] -= m[0] / m[1] * epsilon
    return np.r_[center(r, m).ravel(), center(case["v"], m).ravel()]


def run_case(key="euler_visible", output="results/presets"):
    """Έξι λύσεις: δύο αρχικές καταστάσεις × τρία επίπεδα ακρίβειας."""
    case = PRESETS[key]
    m, duration = case["masses"], case["duration"]
    times = np.linspace(0, duration, 3001)
    settings = [(1e-11, 1e-13, 0.03), (1e-12, 1e-14, 0.015), (1e-13, 1e-15, 0.0075)]
    if key == "burrau":
        settings = [(1e-12, 1e-14, 0.01), (1e-13, 1e-15, 0.005), (3e-14, 3e-16, 0.0025)]
    samples, report = [], dict(
        case=key,
        label=case["label"],
        duration=duration,
        epsilon=1e-6,
        masses=m.tolist(),
        r=case["r"].tolist(),
        v=case["v"].tolist(),
        levels=[],
    )
    for rt, at, step in settings:
        sols = [
            integrate(preset_initial(case, p), rt, at, step, m, duration)
            for p in (False, True)
        ]
        samples.append([s.sol(times).T for s in sols])
        report["levels"].append(
            dict(
                rtol=rt,
                atol=at,
                max_step=step,
                A=diagnostics(sols[0], times, m),
                B=diagnostics(sols[1], times, m),
            )
        )
    a, b = samples[-1]
    delta = distance(a, b, m)
    errors = [
        np.maximum(distance(s[0], a, m), distance(s[1], b, m)) for s in samples[:-1]
    ]
    report["initial_distance"] = float(delta[0])
    report["final_amplification"] = float(delta[-1] / delta[0])
    report["maximum_amplification"] = float(delta.max() / delta[0])
    report["base_reference_difference"] = float(errors[0].max())
    report["strict_reference_difference"] = float(errors[1].max())
    report["minimum_signal_ratio"] = float(
        np.min(delta / np.maximum(errors[0] + errors[1], 1e-30))
    )
    report["comparison_floor"] = 1e-10
    report["differences_below_floor"] = bool(
        max(errors[0].max(), errors[1].max()) < report["comparison_floor"]
    )
    # Η μικρή απόκλιση είναι αποδεκτή: οι περιοδικές λύσεις δεν χρειάζεται
    # να περάσουν το κριτήριο ενίσχυσης που είχε το πείραμα Burrau.
    report["numerical_checks_passed"] = bool(
        all(
            d["relative_energy_drift"] < 1e-7
            and d["absolute_momentum_drift"] < 1e-8
            and d["absolute_angular_momentum_drift"] < 1e-8
            for level in report["levels"]
            for d in (level["A"], level["B"])
        )
        and (errors[1].max() < errors[0].max() or report["differences_below_floor"])
        and report["minimum_signal_ratio"] > 100
    )
    if case["omega"] is not None:
        angle = case["omega"] * times
        r0 = center(case["r"], m)
        expected = np.empty((len(times), 3, 2))
        expected[:, :, 0] = (
            np.cos(angle)[:, None] * r0[:, 0] - np.sin(angle)[:, None] * r0[:, 1]
        )
        expected[:, :, 1] = (
            np.sin(angle)[:, None] * r0[:, 0] + np.cos(angle)[:, None] * r0[:, 1]
        )
        report["max_error_against_exact_rotation"] = float(
            np.max(np.linalg.norm(a[:, :6].reshape(-1, 3, 2) - expected, axis=2))
        )
        acc = rhs(0, preset_initial(case), m)[6:].reshape(3, 2)
        report["initial_rotation_residual"] = float(
            np.max(abs(acc + case["omega"] ** 2 * r0))
        )
        report["numerical_checks_passed"] = bool(
            report["numerical_checks_passed"]
            and report["max_error_against_exact_rotation"] < 1e-7
            and report["initial_rotation_residual"] < 1e-12
        )
    out = Path(output) / key
    out.mkdir(parents=True, exist_ok=True)
    (out / "checks.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    np.savez_compressed(out / "trajectories.npz", t=times, A=a, B=b, delta=delta)
    plot_case(times, a, b, delta, errors, case, out)
    if not report["numerical_checks_passed"]:
        raise RuntimeError(
            "Οι αριθμητικοί έλεγχοι δεν πέρασαν. Δείτε " + str(out / "checks.json")
        )
    return times, a, b, report


def plot_case(t, a, b, delta, errors, case, out):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    colors = ["#0072B2", "#D55E00", "#009E73"]
    for i, color in enumerate(colors):
        ax[0].plot(a[:, 2 * i], a[:, 2 * i + 1], color=color, lw=1, label=f"Σώμα {i+1}")
        ax[0].plot(b[:, 2 * i], b[:, 2 * i + 1], color=color, ls="--", lw=1)
        ax[0].scatter(a[0, 2 * i], a[0, 2 * i + 1], color=color, s=25)
    # Ένωση των τελικών θέσεων: φαίνεται το τρίγωνο ή η ευθεία.
    end = a[-1, :6].reshape(3, 2)[[0, 1, 2, 0]]
    ax[0].plot(end[:, 0], end[:, 1], ":", color=".45", lw=1)
    ax[0].set(
        title="Τροχιές: Α συνεχής, Β διακεκομμένη",
        xlabel="x",
        ylabel="y",
        aspect="equal",
    )
    ax[0].legend(fontsize=8)
    r = a[:, :6].reshape(-1, 3, 2)
    for i, j in PAIRS:
        sep = np.linalg.norm(r[:, i] - r[:, j], axis=1)
        ax[1].plot(t, sep / sep[0], label=f"Σώματα {i+1}–{j+1}")
    ax[1].set(
        title="Αλλάζουν οι αποστάσεις στην Α;",
        xlabel="Χρόνος",
        ylabel="Απόσταση / αρχική απόσταση",
    )
    if case["omega"] is not None:
        ax[1].set_ylim(
            0.98, 1.02
        )  # Δεν μεγεθύνουμε παραπλανητικά τον αριθμητικό θόρυβο.
    ax[1].legend(fontsize=8)
    ax[2].semilogy(t, delta, color="k", label="Διαφορά Α–Β")
    for err, label in zip(errors, ["Βασική – αναφοράς", "Αυστηρή – αναφοράς"]):
        ax[2].semilogy(t, np.maximum(err, 1e-16), lw=0.8, label=label)
    ax[2].set(
        title="Μικρή αρχική αλλαγή και εξέλιξη", xlabel="Χρόνος", ylabel="Απόσταση D"
    )
    ax[2].legend(fontsize=8)
    for axis in ax[1:]:
        axis.grid(alpha=0.2)
    fig.savefig(out / "comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Εκτέλεση έτοιμης περίπτωσης τριών σωμάτων"
    )
    parser.add_argument(
        "case", choices=list(PRESETS), nargs="?", default="euler_visible"
    )
    args = parser.parse_args()
    _, _, _, checks = run_case(args.case)
    print(json.dumps(checks, ensure_ascii=False, indent=2))
