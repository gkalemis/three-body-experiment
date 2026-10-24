"""Διαδραστικό πείραμα Α–Β. Standalone HTML animation χωρίς εξωτερικές πηγές."""

from pathlib import Path
import json
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from three_body import center, rhs, distance, diagnostics, invariants, PAIRS, R_STOP
from presets import PRESETS

PLAYER_TEMPLATE = Path(__file__).with_name("player.html").read_text()

LIMITS = dict(
    euler_visible=12.0,
    euler_long=12.0,
    euler=12.0,
    lagrange_stable=60.0,
    lagrange_equal=12.0,
    at_rest=3.0,
    burrau=20.0,
)


def initial_pair(key, fraction=1e-6, direction="x"):
    """Build independent A/B states with balanced initial position shifts."""
    if key not in PRESETS or direction not in ("x", "y"):
        raise ValueError("Άγνωστη διάταξη ή κατεύθυνση.")
    if not np.isfinite(fraction) or not 0 <= fraction <= 0.1:
        raise ValueError("Η σχετική αλλαγή πρέπει να είναι από 0 έως 0,1.")
    c = PRESETS[key]
    m = c["masses"].copy()
    r = center(c["r"], m)
    v = center(c["v"], m)
    # Ελάχιστη αρχική απόσταση: ίδιο νόημα του slider σε όλα τα σετ.
    length = min(np.linalg.norm(r[i] - r[j]) for i, j in PAIRS)
    shift = np.zeros_like(r)
    # Κανονικοποίηση ώστε η μεγαλύτερη από τις δύο μετατοπίσεις να είναι fraction*length.
    axis = 0 if direction == "x" else 1
    ratio = m[0] / m[1]
    shift[0, axis] = fraction * length / max(1.0, ratio)
    shift[1, axis] = -ratio * shift[0, axis]
    a = np.r_[r.ravel(), v.ravel()]
    b = np.r_[(r + shift).ravel(), v.ravel()]
    return (
        a,
        b,
        dict(
            characteristic_length=float(length),
            fraction=float(fraction),
            max_shift=float(np.max(np.linalg.norm(shift, axis=1))),
            shifts=shift.tolist(),
            com_difference=float(
                np.linalg.norm(np.sum(m[:, None] * shift, axis=0) / m.sum())
            ),
        ),
    )


def solve_one(y0, masses, end, rtol, atol, step):
    """Integrate until the requested end or a close-approach event."""
    events = []
    for i, j in PAIRS:

        def close(t, y, i=i, j=j):
            r = y[:6].reshape(3, 2)
            return np.linalg.norm(r[i] - r[j]) - R_STOP

        close.terminal = True
        close.direction = -1
        events.append(close)
    sol = solve_ivp(
        lambda t, y: rhs(t, y, masses),
        (0, end),
        y0,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        max_step=step,
        dense_output=True,
        events=events,
    )
    if not sol.success:
        raise RuntimeError(f"Αποτυχία υπολογισμού στο t={sol.t[-1]:.8g}: {sol.message}")
    return sol


def compare(key="euler_visible", fraction=1e-6, direction="x", end=None, output=None):
    c = PRESETS[key]
    end = float(c["duration"] if end is None else end)
    if not np.isfinite(end) or not 0.05 <= end <= LIMITS[key]:
        raise ValueError(
            f"Ο τελικός χρόνος πρέπει να είναι από 0,05 έως {LIMITS[key]:g}."
        )
    ya, yb, meta = initial_pair(key, fraction, direction)
    m = c["masses"]
    settings = [(1e-11, 1e-13, 0.03), (1e-12, 1e-14, 0.015), (1e-13, 1e-15, 0.0075)]
    if key == "burrau":
        settings = [(1e-12, 1e-14, 0.01), (1e-13, 1e-15, 0.005), (3e-14, 3e-16, 0.0025)]
    solutions = []
    for rt, at, step in settings:
        sa = solve_one(ya, m, end, rt, at, step)
        sb = sa if fraction == 0 else solve_one(yb, m, end, rt, at, step)
        solutions.append((sa, sb))
    # Κοινό πραγματικό διάστημα και για τις έξι λύσεις. Ποτέ extrapolation μετά από event.
    actual = min(s.t[-1] for pair in solutions for s in pair)
    times = np.linspace(0, actual, 3001)
    samples = [(sa.sol(times).T, sb.sol(times).T) for sa, sb in solutions]
    a, b = samples[-1]
    delta = distance(a, b, m)
    errors = [np.maximum(distance(x, a, m), distance(y, b, m)) for x, y in samples[:-1]]
    levels = [
        dict(
            rtol=rt,
            atol=at,
            max_step=step,
            A=diagnostics(sa, times, m),
            B=diagnostics(sb, times, m),
        )
        for (rt, at, step), (sa, sb) in zip(settings, solutions)
    ]
    # Περιλαμβάνει κλιμάκωση για πολύ μεγάλες τροχιές, χωρίς να κρύβει τον λόγο σήματος/σφάλματος.
    scale = max(
        1.0,
        float(np.max(np.linalg.norm(a[:, :6].reshape(-1, 3, 2), axis=2))),
        float(np.max(np.linalg.norm(b[:, :6].reshape(-1, 3, 2), axis=2))),
    )
    err0, err1 = (float(e.max()) for e in errors)
    convergence = err1 <= err0 * 1.1 or max(err0, err1) < 1e-10 * scale
    invariant_ok = all(
        d["relative_energy_drift"] < 1e-7
        and d["absolute_momentum_drift"] < 1e-8
        and d["absolute_angular_momentum_drift"] < 1e-8
        and d["com_drift"] < 1e-8 * scale
        for level in levels
        for d in (level["A"], level["B"])
    )
    signal_ratio = float(delta[-1] / max(err0 + err1, 1e-30)) if fraction else None
    resolved = bool(fraction == 0 or signal_ratio > 100)
    report = dict(
        case=key,
        requested_end=end,
        actual_end=float(actual),
        stopped_early=bool(actual < end - 1e-10),
        direction=direction,
        **meta,
        initial_distance=float(delta[0]),
        final_distance=float(delta[-1]),
        maximum_distance=float(delta.max()),
        final_amplification=float(delta[-1] / delta[0]) if delta[0] > 0 else None,
        base_reference_difference=err0,
        strict_reference_difference=err1,
        final_signal_error_ratio=signal_ratio,
        convergence_passed=bool(convergence),
        invariants_passed=bool(invariant_ok),
        difference_resolved=resolved,
        numerical_checks_passed=bool(convergence and invariant_ok and resolved),
        levels=levels,
    )
    if c["omega"] is not None:
        r = center(c["r"], m)
        angle = c["omega"] * times
        expected = np.stack(
            (
                np.cos(angle)[:, None] * r[:, 0] - np.sin(angle)[:, None] * r[:, 1],
                np.sin(angle)[:, None] * r[:, 0] + np.cos(angle)[:, None] * r[:, 1],
            ),
            axis=2,
        )
        report["exact_A_error"] = float(
            np.max(np.linalg.norm(a[:, :6].reshape(-1, 3, 2) - expected, axis=2))
        )
        report["numerical_checks_passed"] &= report["exact_A_error"] < 1e-7
    result = dict(t=times, A=a, B=b, delta=delta, errors=errors, report=report, key=key)
    if output:
        out = Path(output)
        out.mkdir(parents=True, exist_ok=True)
        (out / "checks.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
        np.savez_compressed(out / "trajectories.npz", t=times, A=a, B=b, delta=delta)
    return result


def preview_png(key, fraction, direction):
    import io

    a, b, meta = initial_pair(key, fraction, direction)
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.84, bottom=0.32)
    colors = ["#0072B2", "#D55E00", "#009E73"]
    for i, color in enumerate(colors):
        ax.plot(
            a[2 * i],
            a[2 * i + 1],
            "o",
            color=color,
            ms=10,
            label=f"Σώμα {i+1}: Α ●, Β ×",
        )
        ax.plot(b[2 * i], b[2 * i + 1], "x", color=color, ms=10, mew=2)
    ax.plot(0, 0, "k+", ms=13, label="Κοινό κέντρο μάζας")
    ax.set(
        xlabel="x", ylabel="y", title="Αρχικές θέσεις — ίδια κλίμακα", aspect="equal"
    )
    # Keep collinear configurations readable without changing the physical data.
    positions = np.concatenate((a[:6].reshape(3, 2), b[:6].reshape(3, 2)))
    low, high = positions.min(axis=0), positions.max(axis=0)
    midpoint = (low + high) / 2
    span = np.maximum(high - low, max(float(np.max(high - low)), 0.1) * 0.5)
    half = span * 0.65
    ax.set_xlim(midpoint[0] - half[0], midpoint[0] + half[0])
    ax.set_ylim(midpoint[1] - half[1], midpoint[1] + half[1])
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 0.01), fontsize=7.5, frameon=False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return buf.getvalue(), meta


def snapshot(result, path):
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.0), layout="constrained")
    arrays = [
        result["A"][:, :6].reshape(-1, 3, 2),
        result["B"][:, :6].reshape(-1, 3, 2),
    ]
    both = np.concatenate(arrays)
    lo = both.min(axis=(0, 1))
    hi = both.max(axis=(0, 1))
    mid = (lo + hi) / 2
    half = max(float(max(hi - lo)) / 2, 0.1) * 1.12
    for ax, r, title in zip(axes[:2], arrays, ["Εκδοχή Α", "Εκδοχή Β"]):
        for i, col in enumerate(["#0072B2", "#D55E00", "#009E73"]):
            ax.plot(r[:, i, 0], r[:, i, 1], color=col, lw=0.8, alpha=0.7)
            ax.plot(r[-1, i, 0], r[-1, i, 1], "o", color=col)
        ax.set(
            xlim=(mid[0] - half, mid[0] + half),
            ylim=(mid[1] - half, mid[1] + half),
            aspect="equal",
            title=title,
            xlabel="x",
            ylabel="y",
        )
    axes[2].plot(result["t"], result["delta"])
    axes[2].set(title="Διαφορά Α–Β", xlabel="Χρόνος μοντέλου", ylabel="D")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def animation_html(result, frames=480):
    if frames not in (240, 480, 720):
        raise ValueError("Επίλεξε 240, 480 ή 720 καρέ.")
    idx = np.unique(np.rint(np.linspace(0, len(result["t"]) - 1, frames)).astype(int))
    data = dict(
        t=result["t"][idx].tolist(),
        a=result["A"][idx, :6].tolist(),
        b=result["B"][idx, :6].tolist(),
        d=result["delta"][idx].tolist(),
        report=result["report"],
    )
    template = PLAYER_TEMPLATE
    return template.replace(
        '"__THREE_BODY_DATA__"',
        json.dumps(data, separators=(",", ":"), allow_nan=False),
    )


def show_experiment(auto_run=False):
    """Display controls, invalidate stale results, and render completed runs."""
    import ipywidgets as w
    from IPython.display import display, HTML, Markdown, Image, FileLink, IFrame

    style = {"description_width": "initial"}
    choice = w.Dropdown(
        options=[(v["label"], k) for k, v in PRESETS.items()],
        value="euler_visible",
        description="Διάταξη Α:",
        style=style,
        layout=w.Layout(width="98%"),
    )
    amount = w.Dropdown(
        options=[
            ("Ίδιες συνθήκες — Α=Β", 0.0),
            ("Πολύ μικρή αλλαγή", 1e-8),
            ("Μικρή αλλαγή — προεπιλογή", 1e-6),
            ("Μεσαία αλλαγή", 1e-4),
            ("Μεγάλη αλλαγή", 1e-2),
            ("Προσαρμογή με slider", -1.0),
        ],
        value=1e-6,
        description="Αλλαγή Β:",
        style=style,
    )
    epsilon = w.FloatLogSlider(
        value=1e-6,
        base=10,
        min=-8,
        max=-1,
        step=0.25,
        readout_format=".2e",
        continuous_update=False,
        description="Σχετική αλλαγή:",
        style=style,
        disabled=True,
        layout=w.Layout(width="95%"),
    )
    direction = w.Dropdown(
        options=[("Οριζόντια", "x"), ("Κατακόρυφη", "y")],
        value="x",
        description="Κατεύθυνση:",
        style=style,
    )
    duration = w.FloatSlider(
        value=8,
        min=0.05,
        max=12,
        step=0.05,
        continuous_update=False,
        description="Τελικός χρόνος:",
        style=style,
        layout=w.Layout(width="70%"),
    )
    endfield = w.BoundedFloatText(
        value=8, min=0.05, max=12, step=0.05, description="t max:", style=style
    )
    time_link = w.link((duration, "value"), (endfield, "value"))
    quality = w.Dropdown(
        options=[
            ("Κανονική — 240 καρέ", 240),
            ("Ομαλότερη — 480 καρέ", 480),
            ("Λεπτομερής — 720 καρέ", 720),
        ],
        value=480,
        description="Προβολή:",
        style=style,
    )
    run = w.Button(description="Υπολόγισε Α και Β", button_style="primary")
    reset = w.Button(description="Επαναφορά σετ")
    movie_button = w.Button(description="Προβολή κίνησης", disabled=True)
    download = w.Button(description="Λήψη HTML", disabled=True)
    preview = w.Output()
    status = w.Output()
    movie = w.Output()
    table = w.Output()
    state = dict(result=None, html=None, busy=False)
    controls = [
        choice,
        amount,
        epsilon,
        direction,
        duration,
        endfield,
        quality,
        run,
        reset,
        movie_button,
        download,
    ]
    guard = [False]

    def fraction():
        return epsilon.value if amount.value == -1 else amount.value

    def refresh(change=None):
        if guard[0]:
            return
        state.update(result=None, html=None)
        movie_button.disabled = download.disabled = True
        movie.clear_output(wait=False)
        status.clear_output(wait=False)
        epsilon.disabled = amount.value != -1
        png, meta = preview_png(choice.value, fraction(), direction.value)
        with preview:
            preview.clear_output(wait=True)
            display(Image(data=png, width=560))
            display(
                Markdown(
                    f"Μεγαλύτερη μετατόπιση: **{meta['max_shift']:.3e}** μονάδες μήκους. Διαφορά κέντρων μάζας: **{meta['com_difference']:.2e}**. Οι ταχύτητες δεν αλλάζουν· η ενέργεια μπορεί να αλλάζει."
                )
            )
        a, b, _ = initial_pair(choice.value, fraction(), direction.value)
        rows = [
            "|Σώμα|Μάζα|x Α|y Α|x Β|y Β|vx|vy|",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for i, m in enumerate(PRESETS[choice.value]["masses"]):
            rows.append(
                "|"
                + str(i + 1)
                + "|"
                + "|".join(
                    f"{x:.10g}"
                    for x in [
                        m,
                        *a[2 * i : 2 * i + 2],
                        *b[2 * i : 2 * i + 2],
                        *a[6 + 2 * i : 8 + 2 * i],
                    ]
                )
                + "|"
            )
        with table:
            table.clear_output(wait=True)
            display(Markdown("\n".join(rows)))

    def defaults(change=None):
        guard[0] = True
        cap = LIMITS[choice.value]
        val = PRESETS[choice.value]["duration"]
        # Αλλάζουμε πρώτα το όριο χωρίς παροδικά μη έγκυρη τιμή.
        duration.value = min(duration.value, cap)
        endfield.value = min(endfield.value, cap)
        duration.max = endfield.max = cap
        duration.value = val
        amount.value = 1e-6
        epsilon.value = 1e-6
        direction.value = "x"
        guard[0] = False
        refresh()

    def animate(event=None):
        if state["result"] is None:
            return
        html = animation_html(state["result"], quality.value)
        state["html"] = html
        out = Path("results/interactive")
        out.mkdir(parents=True, exist_ok=True)
        path = out / "motion.html"
        path.write_text(html)
        with movie:
            movie.clear_output(wait=True)
            # Απομονωμένο iframe: επανειλημμένες προβολές δεν μοιράζονται timers ή IDs.
            import html as html_module

            display(
                IFrame(
                    src="about:blank", width="100%", height=650,
                    extras=[
                        'title="Κίνηση Α και Β"',
                        'style="border:0"',
                        'srcdoc="' + html_module.escape(html, quote=True) + '"',
                    ],
                )
            )
            display(FileLink(str(path), result_html_prefix="Λήψη αυτόνομης κίνησης: "))

    def execute(event=None):
        if state["busy"]:
            return
        state.update(busy=True, result=None, html=None)
        movie.clear_output(wait=False)
        for ctl in controls:
            ctl.disabled = True
        with status:
            status.clear_output(wait=True)
            display(Markdown("Υπολογίζονται Α και Β σε **τρία επίπεδα ακρίβειας**…"))
        try:
            result = compare(
                choice.value,
                fraction(),
                direction.value,
                duration.value,
                output="results/interactive",
            )
            state["result"] = result
            r = result["report"]
            snapshot(result, "results/interactive/comparison.png")
            label = (
                "ικανοποιήθηκαν"
                if r["numerical_checks_passed"]
                else "ΔΕΝ ικανοποιήθηκαν — η εικόνα είναι διερευνητική"
            )
            with status:
                status.clear_output(wait=True)
                display(
                    Markdown(
                        f"**Έλεγχοι: {label}.** Ζητήθηκε t={r['requested_end']:g}, υπολογίστηκε έως **t={r['actual_end']:.8g}**. Τελική διαφορά D={r['final_distance']:.4g}."
                        + (
                            " **Διακοπή λόγω πολύ κοντινής προσέγγισης· δεν προβάλλεται μεταγενέστερη κίνηση.**"
                            if r["stopped_early"]
                            else ""
                        )
                        + " Η απόκλιση από μόνη της δεν αποδεικνύει χάος."
                    )
                )
                display(Image(filename="results/interactive/comparison.png", width=760))
                display(
                    FileLink(
                        "results/interactive/checks.json",
                        result_html_prefix="Αναλυτικοί έλεγχοι: ",
                    )
                )
            animate()
        except Exception as exc:
            state["result"] = None
            with status:
                status.clear_output(wait=True)
                display(Markdown("**Ο υπολογισμός δεν ολοκληρώθηκε:** " + str(exc)))
        finally:
            state["busy"] = False
            for ctl in controls:
                ctl.disabled = False
            epsilon.disabled = amount.value != -1
            movie_button.disabled = download.disabled = state["result"] is None

    def download_html(event=None):
        if state["result"] is None:
            return
        if state["html"] is None:
            animate()
        path = Path("results/interactive/motion.html")
        try:
            from google.colab import files
        except ImportError:
            with movie:
                display(FileLink(str(path), result_html_prefix="Αποθήκευση HTML: "))
        else:
            files.download(str(path))

    download.on_click(download_html)

    def quality_changed(change):
        if state["result"] is not None:
            movie.clear_output(wait=False)
            state["html"] = None

    choice.observe(defaults, names="value")
    for ctl in [amount, epsilon, direction, duration]:
        ctl.observe(refresh, names="value")
    quality.observe(quality_changed, names="value")
    run.on_click(execute)
    reset.on_click(defaults)
    movie_button.on_click(animate)
    display(
        w.VBox(
            [
                choice,
                amount,
                epsilon,
                direction,
                w.HBox([duration, endfield]),
                quality,
                w.HBox([run, reset, movie_button, download]),
                preview,
                table,
                status,
                movie,
            ]
        )
    )
    refresh()
    if auto_run:
        execute()
    return dict(
        choice=choice,
        amount=amount,
        epsilon=epsilon,
        direction=direction,
        duration=duration,
        endfield=endfield,
        quality=quality,
        run=run,
        reset=reset,
        movie_button=movie_button,
        state=state,
        execute=execute,
        animate=animate,
        refresh=refresh,
        time_link=time_link,
        download=download,
    )
