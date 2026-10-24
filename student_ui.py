"""Student interface; the shared numerical engine and checks remain unchanged."""
from pathlib import Path
import html
from experiment import PRESETS, LIMITS, compare, animation_html, preview_png


def student_animation(result, frames=480):
    """Reuse the checked player with a simpler visible clock and optional chart."""
    page = animation_html(result, frames)
    full_clock = (
        'Χρόνος μοντέλου t = ${data.t[k].toFixed(5)} / '
        '${data.t[n - 1].toFixed(5)} μονάδες χρόνου | '
        'Διαφορά D = ${data.d[k].toExponential(3)} | Καρέ ${k + 1}/${n}'
    )
    simple_clock = (
        'Χρόνος: ${data.t[k].toFixed(2)} / '
        '${data.t[n - 1].toFixed(2)} μονάδες χρόνου'
    )
    if full_clock not in page:
        raise ValueError('Το πρότυπο του player άλλαξε· χρειάζεται ενημέρωση της μαθητικής προβολής.')
    page = page.replace(full_clock, simple_clock)
    page = page.replace(
        '<h3>Α και Β: κοινός χρόνος, ίδια κλίμακα</h3>',
        '<h3>Παρατήρησε τις δύο κινήσεις</h3>'
    )
    # The chart remains available on demand, without changing its data or logic.
    chart_start = page.index('  <canvas\n    id="chart"')
    chart_end = page.index('</canvas>', chart_start) + len('</canvas>')
    page = (
        page[:chart_start]
        + '<details><summary>Θέλω να δω και το διάγραμμα της διαφοράς</summary>'
        + page[chart_start:chart_end] + '</details>' + page[chart_end:]
    )
    return page


def show_student_experiment(auto_run=True):
    """A few visible choices, animation first, technical information folded away."""
    import ipywidgets as w
    from IPython.display import display, HTML, Image, FileLink, IFrame

    style = {'description_width': 'initial'}
    wide = w.Layout(width='96%')
    labels = {
        'euler_visible': 'Ξεκίνα εδώ: τρία σώματα σε ευθεία (χρόνος 8)',
        'euler_long': 'Ίδια ευθεία, περισσότερος χρόνος (12)',
        'lagrange_stable': 'Περιστρεφόμενο τρίγωνο με άνισες μάζες',
        'lagrange_equal': 'Περιστρεφόμενο τρίγωνο με ίσες μάζες',
        'euler': 'Ίδια ευθεία, λιγότερος χρόνος (6)',
        'at_rest': 'Τρίγωνο που ξεκινά ακίνητο',
        'burrau': 'Τρία ακίνητα σώματα με διαφορετικές μάζες',
    }
    choice = w.Dropdown(
        options=[(labels[key], key) for key in PRESETS], value='euler_visible',
        description='1. Διάλεξε διάταξη:', style=style, layout=wide,
    )
    change = w.Dropdown(
        options=[('Καμία — οι δύο εκδοχές είναι ίδιες', 0.),
                 ('Πολύ μικρή', 1e-8), ('Μικρή — προτείνεται', 1e-6),
                 ('Μεσαία', 1e-4), ('Μεγαλύτερη', .01)],
        value=1e-6, description='2. Άλλαξε λίγο τη Β:', style=style, layout=wide,
    )
    duration = w.FloatSlider(
        value=8, min=.05, max=12, step=.05, continuous_update=False,
        description='3. Μέχρι ποιον χρόνο;', style=style, layout=wide,
    )
    endfield = w.BoundedFloatText(
        value=8, min=.05, max=12, description='Ή γράψε χρόνο:', style=style,
    )
    time_link = w.link((duration, 'value'), (endfield, 'value'))
    direction = w.Dropdown(
        options=[('Οριζόντια', 'x'), ('Κατακόρυφη', 'y')], value='x',
        description='Κατεύθυνση αλλαγής:', style=style, layout=wide,
    )
    quality = w.Dropdown(
        options=[('Κανονική', 240), ('Ομαλότερη — προτείνεται', 480),
                 ('Περισσότερα καρέ', 720)], value=480,
        description='Κίνηση:', style=style, layout=wide,
    )
    run = w.Button(description='Δες τι θα συμβεί', button_style='primary')
    reset = w.Button(description='Ξεκίνα από την αρχή')
    download = w.Button(description='Κράτησε την κίνηση', disabled=True)
    message = w.HTML()
    movie = w.Output()
    preview = w.Output()
    details = w.Output()
    extras = w.Accordion(children=[w.VBox([direction, quality, preview]), details])
    extras.set_title(0, 'Περισσότερες επιλογές και αρχικές θέσεις')
    extras.set_title(1, 'Για όποιον θέλει: αριθμοί και έλεγχοι')
    extras.selected_index = None
    state = dict(result=None, html=None, busy=False)
    guarded = [False]
    controls = [choice, change, duration, endfield, direction, quality, run, reset, download]

    def draw_preview():
        png, _ = preview_png(choice.value, change.value, direction.value)
        with preview:
            preview.clear_output(wait=True)
            display(Image(data=png))

    def invalidate(event=None):
        if guarded[0]:
            return
        state.update(result=None, html=None)
        download.disabled = True
        movie.clear_output(wait=False)
        details.clear_output(wait=False)
        message.value = 'Οι επιλογές άλλαξαν. Πάτησε <b>Δες τι θα συμβεί</b>.'
        draw_preview()

    def reset_case(event=None):
        guarded[0] = True
        cap = LIMITS[choice.value]
        duration.value = min(duration.value, cap)
        endfield.value = min(endfield.value, cap)
        duration.max = endfield.max = cap
        duration.value = PRESETS[choice.value]['duration']
        change.value = 1e-6
        direction.value = 'x'
        guarded[0] = False
        invalidate()

    def render():
        if state['result'] is None:
            return
        page = student_animation(state['result'], quality.value)
        state['html'] = page
        folder = Path('results/students')
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / 'motion.html'
        path.write_text(page, encoding='utf-8')
        with movie:
            movie.clear_output(wait=True)
            display(IFrame(
                src="about:blank", width="100%", height=820,
                extras=[
                    'title="Παρατήρησε Α και Β"',
                    'style="border:0"',
                    'srcdoc="' + html.escape(page, quote=True) + '"',
                ],
            ))

    def execute(event=None):
        if state['busy']:
            return
        state.update(busy=True, result=None, html=None)
        for control in controls:
            control.disabled = True
        movie.clear_output(wait=False)
        message.value = 'Ετοιμάζονται οι δύο κινήσεις. Περίμενε λίγο…'
        try:
            result = compare(
                choice.value, change.value, direction.value, duration.value,
                output='results/students',
            )
            state['result'] = result
            report = result['report']
            if not report['numerical_checks_passed']:
                message.value = (
                    '<b>Προσοχή:</b> οι έλεγχοι ακρίβειας δεν ικανοποιήθηκαν. '
                    'Η εικόνα είναι διερευνητική. Δοκίμασε μικρότερη διάρκεια.'
                )
            elif report['stopped_early']:
                message.value = (
                    'Δύο σώματα πλησίασαν πολύ. Σταματήσαμε στον χρόνο '
                    f"<b>{report['actual_end']:.4f}</b>, πριν από τον χρόνο που ζήτησες."
                )
            else:
                message.value = '<b>Έτοιμο!</b> Πάτησε ▶ και σύγκρινε τις δύο κινήσεις.'
            with details:
                details.clear_output(wait=True)
                display(HTML(
                    f"<p>Τελική διαφορά D: {report['final_distance']:.5g}. "
                    f"Ίδιο αρχικό κέντρο μάζας: διαφορά {report['com_difference']:.2e}.</p>"
                ))
                display(FileLink('results/students/checks.json',
                                 result_html_prefix='Αναλυτικοί έλεγχοι: '))
            render()
        except Exception as exc:
            state.update(result=None, html=None)
            message.value = (
                'Δεν ολοκληρώθηκε η δοκιμή. Επίλεξε μικρότερη διάρκεια ή άλλη διάταξη.'
            )
            with details:
                details.clear_output(wait=True)
                display(HTML('<p>' + html.escape(str(exc)) + '</p>'))
        finally:
            state['busy'] = False
            for control in controls:
                control.disabled = False
            download.disabled = state['result'] is None

    def quality_changed(event=None):
        # Reuse existing solutions, never reintegrate for a display-only change.
        if state['result'] is not None:
            render()

    def save(event=None):
        if state['html'] is None:
            return
        path = Path('results/students/motion.html')
        try:
            from google.colab import files
        except ImportError:
            with movie:
                display(FileLink(str(path), result_html_prefix='Αποθήκευση: '))
        else:
            files.download(str(path))

    choice.observe(reset_case, names='value')
    for control in [change, duration, direction]:
        control.observe(invalidate, names='value')
    quality.observe(quality_changed, names='value')
    run.on_click(execute)
    reset.on_click(reset_case)
    download.on_click(save)
    display(w.VBox([
        choice, change, duration, endfield,
        w.HBox([run, reset, download]), message, movie, extras,
    ]))
    draw_preview()
    if auto_run:
        execute()
    return dict(
        choice=choice, change=change, duration=duration, endfield=endfield,
        direction=direction, quality=quality, run=run, reset=reset,
        download=download, state=state, extras=extras, message=message,
        execute=execute, time_link=time_link,
    )
