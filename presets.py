"""Έξι περιπτώσεις, μενού και κινούμενη απεικόνιση στο notebook.

Τοπική αριθμητική εκτέλεση: python presets.py euler.
Στο notebook: Εκτέλεση περίπτωσης → Προβολή κίνησης → αναπαραγωγή.
Η motion_html χρησιμοποιεί τις έτοιμες λύσεις και δεν ξαναλύνει τις εξισώσεις.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from three_body import center, rhs, integrate, distance, diagnostics, PAIRS


def make_presets():
    """Κάθε πακέτο δίνει μάζες, θέσεις, ταχύτητες και διάρκεια."""
    triangle = np.array([[0., 2/np.sqrt(3)], [-1., -1/np.sqrt(3)],
                         [1., -1/np.sqrt(3)]])  # Πλευρά 2.

    def rotating(label, masses, positions, omega, duration, explanation):
        m = np.array(masses, dtype=float)
        r = center(positions, m)
        v = omega * np.column_stack((-r[:, 1], r[:, 0]))
        return dict(label=label, masses=m, r=r, v=v, omega=omega,
                    duration=duration, explanation=explanation)

    equal = rotating('Ισόπλευρο τρίγωνο — ίσες μάζες', [1, 1, 1],
        triangle, np.sqrt(3/8), 10.,
        'Το τρίγωνο περιστρέφεται με σταθερό σχήμα και μέγεθος. '
        'Είναι σχετική ισορροπία: φαίνεται ακίνητο αν περιστρεφόμαστε μαζί του. '
        'Για ίσες μάζες είναι ασταθές· μια μικρή διαταραχή μπορεί να ενισχυθεί. '
        'Αυτό δεν σημαίνει ότι θα φανεί μεγάλη αλλαγή σε τόσο μικρή διάρκεια.')
    unequal = rotating('Ισόπλευρο τρίγωνο — ένα βαρύ σώμα', [.001, .01, 1],
        triangle, np.sqrt(1.011/8), 30.,
        'Και εδώ το τρίγωνο διατηρεί σχήμα και μέγεθος καθώς περιστρέφεται. '
        'Η συγκεκριμένη αναλογία μαζών ικανοποιεί το κλασικό κριτήριο '
        'γραμμικής ευστάθειας στο επίπεδο. Αυτό αφορά απειροστά μικρές '
        'διαταραχές και δεν εγγυάται κάθε κίνηση για πάντα.')
    line = rotating('Ευθεία Euler — περιστρεφόμενη διάταξη', [1, 1, 1],
        np.array([[-1., 0.], [0., 0.], [1., 0.]]), np.sqrt(1.25), 6.,
        'Το μεσαίο σώμα μένει στο κέντρο μάζας και τα άλλα δύο '
        'περιστρέφονται διατηρώντας την ευθεία. Είναι ασταθής σχετική '
        'ισορροπία. Οι βαρυτικές δυνάμεις στα εξωτερικά σώματα δεν μηδενίζονται: '
        'δίνουν την επιτάχυνση της κυκλικής κίνησης.')
    rest = dict(label='Τρίγωνο αρχικά ακίνητο — αρχίζει να συρρικνώνεται',
        masses=np.ones(3), r=triangle.copy(), v=np.zeros((3, 2)),
        omega=None, duration=1.4,
        explanation='Ίδια γεωμετρία, αλλά χωρίς αρχική περιστροφή. '
        'Τα σώματα ξεκινούν ακίνητα και έλκονται προς το κέντρο: '
        'δεν βρίσκονται σε στατική ισορροπία. Παρακολουθούμε μόνο '
        'το διάστημα πριν από τη σύγκρουση. Δεν συνεχίζουμε μέσα από αυτήν.')
    burrau = dict(label='Burrau — ενίσχυση μικρής αρχικής διαφοράς',
        masses=np.array([3., 4., 5.]),
        r=np.array([[1., 3.], [-2., -1.], [1., -1.]]),
        v=np.zeros((3, 2)), omega=None, duration=15.8,
        explanation='Το αρχικό πείραμα του notebook. Διαδοχικές στενές '
        'προσεγγίσεις ενισχύουν τη μικρή αρχική διαφορά στο επιλεγμένο '
        'χρονικό διάστημα. Η απόκλιση μόνη της δεν αποδεικνύει χάος.')
    visible = dict(line)
    visible.update(label='Ξεκίνα εδώ — εμφανής απόκλιση Euler (t=8)',
        duration=8., animation_interval_ms=250,
        explanation='Ίδιες μάζες, αρχικές θέσεις, ταχύτητες και διαταραχή 10⁻⁶ '
        'με την περίπτωση Euler t=6, αλλά παρακολουθούμε μέχρι t=8. '
        'Η μεγαλύτερη διάρκεια αφήνει την αστάθεια να γίνει ορατή. '
        'Στην Α η ευθεία περιστρέφεται, ενώ η Β απομακρύνεται εμφανώς '
        'από αυτή τη διάταξη. Παρατήρησε ιδιαίτερα το διάστημα t=6 έως t=8. '
        'Η αρχική αναπαραγωγή διαρκεί περίπου 45 δευτερόλεπτα. '
        'Η απόκλιση από ασταθή σχετική ισορροπία δεν αποδεικνύει από μόνη της χάος.')
    return dict(euler_visible=visible, lagrange_stable=unequal,
                lagrange_equal=equal, euler=line, at_rest=rest, burrau=burrau)


PRESETS = make_presets()


def preset_initial(case, perturbed=False, epsilon=1e-6):
    m = case['masses']
    r = case['r'].copy()
    if perturbed:
        r[0, 0] += epsilon
        r[1, 0] -= m[0]/m[1]*epsilon
    return np.r_[center(r, m).ravel(), center(case['v'], m).ravel()]


def run_case(key='euler_visible', output='results/presets'):
    """Έξι λύσεις: δύο αρχικές καταστάσεις × τρία επίπεδα ακρίβειας."""
    case = PRESETS[key]
    m, duration = case['masses'], case['duration']
    times = np.linspace(0, duration, 3001)
    settings = [(1e-11, 1e-13, .03), (1e-12, 1e-14, .015),
                (1e-13, 1e-15, .0075)]
    if key == 'burrau':
        settings = [(1e-12, 1e-14, .01), (1e-13, 1e-15, .005),
                    (3e-14, 3e-16, .0025)]
    samples, report = [], dict(case=key, label=case['label'], duration=duration,
        epsilon=1e-6, masses=m.tolist(), r=case['r'].tolist(), v=case['v'].tolist(),
        levels=[])
    for rt, at, step in settings:
        sols = [integrate(preset_initial(case, p), rt, at, step, m, duration)
                for p in (False, True)]
        samples.append([s.sol(times).T for s in sols])
        report['levels'].append(dict(rtol=rt, atol=at, max_step=step,
            A=diagnostics(sols[0], times, m), B=diagnostics(sols[1], times, m)))
    a, b = samples[-1]
    delta = distance(a, b, m)
    errors = [np.maximum(distance(s[0], a, m), distance(s[1], b, m))
              for s in samples[:-1]]
    report['initial_distance'] = float(delta[0])
    report['final_amplification'] = float(delta[-1]/delta[0])
    report['maximum_amplification'] = float(delta.max()/delta[0])
    report['base_reference_difference'] = float(errors[0].max())
    report['strict_reference_difference'] = float(errors[1].max())
    report['minimum_signal_ratio'] = float(np.min(delta/np.maximum(errors[0]+errors[1], 1e-30)))
    report['comparison_floor'] = 1e-11
    report['differences_below_floor'] = bool(max(errors[0].max(), errors[1].max()) < 1e-11)
    # Η μικρή απόκλιση είναι αποδεκτή: οι περιοδικές λύσεις δεν χρειάζεται
    # να περάσουν το κριτήριο ενίσχυσης που είχε το πείραμα Burrau.
    report['numerical_checks_passed'] = bool(all(
        d['relative_energy_drift'] < 1e-7 and
        d['absolute_momentum_drift'] < 1e-8 and
        d['absolute_angular_momentum_drift'] < 1e-8
        for level in report['levels'] for d in (level['A'], level['B']))
        and (errors[1].max() < errors[0].max() or report['differences_below_floor'])
        and report['minimum_signal_ratio'] > 100)
    if case['omega'] is not None:
        angle = case['omega']*times
        r0 = center(case['r'], m)
        expected = np.empty((len(times), 3, 2))
        expected[:, :, 0] = np.cos(angle)[:, None]*r0[:, 0] - np.sin(angle)[:, None]*r0[:, 1]
        expected[:, :, 1] = np.sin(angle)[:, None]*r0[:, 0] + np.cos(angle)[:, None]*r0[:, 1]
        report['max_error_against_exact_rotation'] = float(np.max(np.linalg.norm(
            a[:, :6].reshape(-1, 3, 2)-expected, axis=2)))
        acc = rhs(0, preset_initial(case), m)[6:].reshape(3, 2)
        report['initial_rotation_residual'] = float(np.max(abs(acc+case['omega']**2*r0)))
    out = Path(output)/key
    out.mkdir(parents=True, exist_ok=True)
    (out/'checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    np.savez_compressed(out/'trajectories.npz', t=times, A=a, B=b, delta=delta)
    plot_case(times, a, b, delta, errors, case, out)
    if not report['numerical_checks_passed']:
        raise RuntimeError('Οι αριθμητικοί έλεγχοι δεν πέρασαν. Δείτε '+str(out/'checks.json'))
    return times, a, b, report


def plot_case(t, a, b, delta, errors, case, out):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    colors = ['#0072B2', '#D55E00', '#009E73']
    for i, color in enumerate(colors):
        ax[0].plot(a[:, 2*i], a[:, 2*i+1], color=color, lw=1, label=f'Σώμα {i+1}')
        ax[0].plot(b[:, 2*i], b[:, 2*i+1], color=color, ls='--', lw=1)
        ax[0].scatter(a[0, 2*i], a[0, 2*i+1], color=color, s=25)
    # Ένωση των τελικών θέσεων: φαίνεται το τρίγωνο ή η ευθεία.
    end = a[-1, :6].reshape(3, 2)[[0, 1, 2, 0]]
    ax[0].plot(end[:, 0], end[:, 1], ':', color='.45', lw=1)
    ax[0].set(title='Τροχιές: Α συνεχής, Β διακεκομμένη', xlabel='x', ylabel='y', aspect='equal')
    ax[0].legend(fontsize=8)
    r = a[:, :6].reshape(-1, 3, 2)
    for i, j in PAIRS:
        sep = np.linalg.norm(r[:, i]-r[:, j], axis=1)
        ax[1].plot(t, sep/sep[0], label=f'Σώματα {i+1}–{j+1}')
    ax[1].set(title='Αλλάζουν οι αποστάσεις στην Α;', xlabel='Χρόνος', ylabel='Απόσταση / αρχική απόσταση')
    if case['omega'] is not None:
        ax[1].set_ylim(.98, 1.02)  # Δεν μεγεθύνουμε παραπλανητικά τον αριθμητικό θόρυβο.
    ax[1].legend(fontsize=8)
    ax[2].semilogy(t, delta, color='k', label='Διαφορά Α–Β')
    for err, label in zip(errors, ['Βασική – αναφοράς', 'Αυστηρή – αναφοράς']):
        ax[2].semilogy(t, np.maximum(err, 1e-16), lw=.8, label=label)
    ax[2].set(title='Μικρή αρχική αλλαγή και εξέλιξη', xlabel='Χρόνος', ylabel='Απόσταση D')
    ax[2].legend(fontsize=8)
    for axis in ax[1:]:
        axis.grid(alpha=.2)
    fig.savefig(out/'comparison.png', dpi=180)
    plt.close(fig)


def motion_html(times, a, b, case, frame_count=180, interval_ms=180):
    """HTML με χειριστήρια αναπαραγωγής, από ήδη υπολογισμένες λύσεις."""
    import re
    from matplotlib.animation import FuncAnimation
    times = np.asarray(times)
    if (len(times) < 2 or a.shape != (len(times), 12) or b.shape != a.shape
            or not np.all(np.diff(times) > 0)):
        raise ValueError('Χρειάζονται δύο λύσεις 12 συνιστωσών σε κοινούς αύξοντες χρόνους.')
    if frame_count < 2 or interval_ms <= 0:
        raise ValueError('Χρειάζονται τουλάχιστον δύο καρέ και θετικό διάστημα προβολής.')
    # Ομοιόμορφες στιγμές εξόδου: επιλέγουμε υποσύνολο μόνο για την εικόνα.
    indices = np.unique(np.rint(np.linspace(0, len(times)-1, frame_count)).astype(int))
    positions = [a[:, :6].reshape(-1, 3, 2), b[:, :6].reshape(-1, 3, 2)]
    both = np.concatenate(positions, axis=0)
    low, high = both.min(axis=(0, 1)), both.max(axis=(0, 1))
    middle = (low+high)/2
    half_width = max(float(np.max(high-low))/2, .1)*1.12
    delta = distance(a, b, case['masses'])
    colors = ['#0072B2', '#D55E00', '#009E73']
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10}):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), dpi=90)
        fig.subplots_adjust(left=.07, right=.98, bottom=.18, top=.78, wspace=.25)
        title = fig.suptitle('', fontsize=11)
        markers, trails = [], []
        for ax, label in zip(axes, ['Εκδοχή Α', 'Εκδοχή Β — μικρή αρχική αλλαγή']):
            ax.set(xlim=(middle[0]-half_width, middle[0]+half_width),
                   ylim=(middle[1]-half_width, middle[1]+half_width),
                   xlabel='x', ylabel='y', title=label, aspect='equal')
            ax.grid(alpha=.15)
            ax.plot(0, 0, '+', color='.5', ms=7, label='Κέντρο μάζας')
            local_markers, local_trails = [], []
            for i, color in enumerate(colors):
                trail, = ax.plot([], [], color=color, alpha=.45, lw=.8)
                marker, = ax.plot([], [], 'o', color=color, ms=7, label=f'Σώμα {i+1}')
                local_markers.append(marker)
                local_trails.append(trail)
            markers.append(local_markers)
            trails.append(local_trails)
        fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center',
                   ncol=4, fontsize=8)

        def update(frame):
            k = indices[frame]
            title.set_text(f"Χρόνος: {times[k]:.3f} / {times[-1]:g}   |   Απόσταση D: {delta[k]:.3e}\n"
                           'Κοινός χρόνος και ίδια κλίμακα — οι γραμμές δείχνουν μόνο το παρελθόν')
            for run, r in enumerate(positions):
                for i in range(3):
                    markers[run][i].set_data([r[k, i, 0]], [r[k, i, 1]])
                    trails[run][i].set_data(r[:k+1, i, 0], r[:k+1, i, 1])
            return [title]+[item for group in markers+trails for item in group]

        animation = FuncAnimation(fig, update, frames=len(indices),
                                  interval=interval_ms, blit=False, repeat=False)
        try:
            html = animation.to_jshtml(default_mode='once')
        finally:
            plt.close(fig)
    # Ελληνικά βοηθητικά κείμενα και σύμβολα χωρίς εξωτερικές γραμματοσειρές.
    translations = {'Decrease speed':'Πιο αργά', 'First frame':'Αρχή',
        'Previous frame':'Προηγούμενο καρέ', 'Play backwards':'Αναπαραγωγή προς τα πίσω',
        'Pause':'Παύση', 'Play':'Αναπαραγωγή', 'Next frame':'Επόμενο καρέ',
        'Last frame':'Τέλος', 'Increase speed':'Πιο γρήγορα',
        'Repetition mode':'Τρόπος επανάληψης'}
    for english, greek in translations.items():
        html = html.replace(f'"{english}"', f'"{greek}"')
    for english, greek in [('Once','Μία φορά'), ('Loop','Επανάληψη'), ('Reflect','Μπρος–πίσω')]:
        html = html.replace('>'+english+'<', '>'+greek+'<')
    icons = {'fa-minus':'−', 'fa-fast-backward':'⏮', 'fa-step-backward':'◀|',
             'fa-play fa-flip-horizontal':'◀', 'fa-pause':'Ⅱ', 'fa-play':'▶',
             'fa-step-forward':'|▶', 'fa-fast-forward':'⏭', 'fa-plus':'+'}
    for icon, symbol in icons.items():
        html = html.replace('<i class="fa '+icon+'"></i>', symbol)
    html = re.sub(r'<link\b[^>]*>', '', html)
    instructions = ('<p lang="el"><b>Παρακολούθησε την κίνηση:</b> πάτησε ▶ για αναπαραγωγή, '
        'Ⅱ για παύση, − για πιο αργά και + για πιο γρήγορα. '
        'Μετακίνησε τη μπάρα για να διαλέξεις στιγμή ή χρησιμοποίησε τα κουμπιά '
        'για ένα καρέ κάθε φορά. Η αρχική προβολή διαρκεί περίπου '
        f'{len(indices)*interval_ms/1000:.1f} δευτερόλεπτα· '
        'ο χρόνος προσομοίωσης φαίνεται πάνω από τα σχήματα.</p>')
    return '<!doctype html><html lang="el"><meta charset="utf-8"><title>Κίνηση τριών σωμάτων</title>'+instructions+html+'</html>'


def show_menu():
    """Μενού με σταθερές περιοχές εμφάνισης, χωρίς σύλληψη εξόδου kernel."""
    import ipywidgets as widgets
    from IPython.display import display, Markdown, Image, HTML
    choice = widgets.Dropdown(options=[(v['label'], k) for k, v in PRESETS.items()],
        value='euler_visible', description='Περίπτωση:',
        style={'description_width': 'initial'}, layout=widgets.Layout(width='95%'))
    button = widgets.Button(description='Εκτέλεση περίπτωσης', button_style='primary',
                            layout=widgets.Layout(width='220px'))
    animate_button = widgets.Button(description='Προβολή κίνησης', disabled=True,
        layout=widgets.Layout(width='220px'))
    display(choice)
    info = display(Markdown(''), display_id=True)
    display(widgets.HBox([button, animate_button]))
    picture = display(Markdown(''), display_id=True)
    summary = display(Markdown(''), display_id=True)
    movie = display(Markdown(''), display_id=True)
    state = dict(last_report=None, last_data=None, animation_html=None)

    def describe(change=None):
        c = PRESETS[choice.value]
        rows = ['| Σώμα | Μάζα | x | y | vx | vy |', '|---|---:|---:|---:|---:|---:|']
        for i in range(3):
            values = [c['masses'][i], *c['r'][i], *c['v'][i]]
            rows.append('| '+str(i+1)+' | '+' | '.join(f'{v:.8g}' for v in values)+' |')
        info.update(Markdown('### '+c['label']+'\n\n'+c['explanation']+'\n\n'+'\n'.join(rows)+
            f"\n\nΔιάρκεια: **{c['duration']:g}** μονάδες χρόνου. Οι τιμές του πίνακα "
            'είναι της Α. Στη Β αλλάζουν μόνο x₁ κατά 10⁻⁶ και x₂ κατά −(m₁/m₂)·10⁻⁶. '
            'Οι ταχύτητες εξασφαλίζουν ακίνητο κέντρο μάζας. Ο πίνακας εμφανίζει '
            '8 σημαντικά ψηφία. Η επιβράδυνση προβολής δεν αυξάνει τον χρόνο προσομοίωσης. '
            'Πάτησε το κουμπί για νέο υπολογισμό.'))
        picture.update(Markdown(''))
        summary.update(Markdown(''))
        state['last_report'] = None
        state['last_data'] = None
        state['animation_html'] = None
        animate_button.disabled = True
        movie.update(Markdown(''))

    def execute(event=None):
        button.disabled = True
        choice.disabled = True
        animate_button.disabled = True
        movie.update(Markdown(''))
        state['animation_html'] = None
        state['last_data'] = None
        state['last_report'] = None
        summary.update(Markdown('**Υπολογίζονται οι δύο εκδοχές σε τρία επίπεδα ακρίβειας…**'))
        try:
            times, a, b, report = run_case(choice.value)
            state['last_data'] = (times, a, b)
            state['last_report'] = report
            picture.update(Image(filename=str(Path('results/presets')/choice.value/'comparison.png')))
            summary.update(Markdown(f"**Τελική ενίσχυση:** {report['final_amplification']:.3g} φορές. "
                f"**Μέγιστη ενίσχυση:** {report['maximum_amplification']:.3g} φορές. "
                '**Αριθμητικοί έλεγχοι: πέρασαν.**\n\n'
                'Μικρή ενίσχυση δεν είναι αποτυχία: οι περιπτώσεις έχουν διαφορετική συμπεριφορά. '
                'Η απόκλιση δεν αποτελεί απόδειξη χάους. Οι αριθμοί του αρχικού κειμένου '
                'του notebook αφορούν μόνο την περίπτωση Burrau.'))
        except Exception as exc:
            picture.update(Markdown(''))
            summary.update(Markdown('**Ο υπολογισμός σταμάτησε:** '+str(exc)))
        finally:
            button.disabled = False
            choice.disabled = False
            animate_button.disabled = state['last_data'] is None

    def animate(event=None):
        if state['last_data'] is None:
            return
        button.disabled = choice.disabled = animate_button.disabled = True
        movie.update(Markdown('**Ετοιμάζονται τα καρέ της κίνησης…**'))
        try:
            case = PRESETS[choice.value]
            html = motion_html(*state['last_data'], case,
                               interval_ms=case.get('animation_interval_ms', 180))
            state['animation_html'] = html
            (Path('results/presets')/choice.value/'motion.html').write_text(html)
            movie.update(HTML(html))
        except Exception as exc:
            movie.update(Markdown('**Δεν δημιουργήθηκε η κινούμενη απεικόνιση:** '+str(exc)))
        finally:
            button.disabled = choice.disabled = animate_button.disabled = False

    animate_button.on_click(animate)
    choice.observe(describe, names='value')
    button.on_click(execute)
    describe()
    execute()  # Η «Εκτέλεση όλων» παράγει αποτέλεσμα χωρίς επιπλέον κλικ.
    return dict(choice=choice, button=button, info=info, picture=picture,
                summary=summary, movie=movie, animate_button=animate_button, state=state)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Εκτέλεση έτοιμης περίπτωσης τριών σωμάτων')
    parser.add_argument('case', choices=list(PRESETS), nargs='?', default='euler_visible')
    args = parser.parse_args()
    _, _, _, checks = run_case(args.case)
    print(json.dumps(checks, ensure_ascii=False, indent=2))
