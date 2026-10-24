"""Build a standalone student edition with collapsed code cells."""
from pathlib import Path
import ast
import hashlib
import nbformat

ROOT = Path(__file__).resolve().parent


def build():
    advanced = nbformat.read(ROOT / 'three_body.ipynb', as_version=4)
    sources = []
    for cell in advanced.cells:
        if cell.cell_type != 'code':
            continue
        tree = ast.parse(cell.source)
        if any(isinstance(node, ast.FunctionDef) and node.name == 'show_experiment'
               for node in tree.body):
            continue
        if cell.source.strip().startswith('controls ='):
            continue
        sources.append(cell.source)
    student_source = (ROOT / 'student_ui.py').read_text()
    sources.append('import html\n')
    for node in ast.parse(student_source).body:
        if isinstance(node, ast.FunctionDef):
            sources.append(ast.get_source_segment(student_source, node))

    intro = '''# Τρία σώματα: παίξε με την αρχή της κίνησης

## Ξεκίνα σε τρία βήματα

1. Πάτησε **Περιβάλλον εκτέλεσης → Εκτέλεση όλων** στο Colab ή **Run → Run All Cells** στο JupyterLab.
2. Περίμενε να εμφανιστεί η εφαρμογή. Δεν χρειάζεται να γράψεις ή να αλλάξεις κώδικα.
3. Πάτησε **▶ Αναπαραγωγή** στην κινούμενη εικόνα.

Η **Α** είναι η αρχική μας αναφορά. Στη **Β** αλλάζουμε λίγο τις αρχικές θέσεις, κρατώντας το ίδιο αρχικό κέντρο μάζας. Με το ίδιο χρώμα βλέπεις το ίδιο σώμα και στις δύο εικόνες. Οι εικόνες έχουν την ίδια κλίμακα.

## Κάνε τη δική σου δοκιμή

Διάλεξε μια διάταξη, πόσο θα αλλάξεις τη Β και μέχρι ποιον χρόνο θα παρακολουθήσεις. Έπειτα πάτησε **Δες τι θα συμβεί**. Ο χρόνος αυτός ανήκει στο μοντέλο· δεν είναι τα δευτερόλεπτα που διαρκεί η προβολή.

- **Χωρίς αλλαγή:** διάλεξε «Καμία». Συμπίπτουν οι δύο κινήσεις;
- **Περισσότερος χρόνος:** στην ευθεία δοκίμασε χρόνο 6, μετά 8 και μετά 12, με την ίδια μικρή αλλαγή. Πότε αρχίζεις να βλέπεις διαφορά;
- **Άλλη αρχή:** σύγκρινε το περιστρεφόμενο τρίγωνο με εκείνο που ξεκινά ακίνητο. Τι αλλάζει;

Στην κίνηση μπορείς να πατήσεις παύση, να διαλέξεις στιγμή, να αλλάξεις ταχύτητα ή να κρύψεις τα ίχνη. Αυτά αλλάζουν μόνο την προβολή. Το διάγραμμα της διαφοράς ανοίγει προαιρετικά κάτω από την εικόνα.

Αν τα σώματα πλησιάσουν υπερβολικά, σταματάμε νωρίτερα και εμφανίζεται μήνυμα. Αν οι έλεγχοι ακρίβειας δεν ικανοποιηθούν, η εικόνα επισημαίνεται ως διερευνητική. Η μεγάλη απόκλιση μόνη της δεν αποδεικνύει χάος.

Ο κώδικας είναι συμπτυγμένος, αλλά παραμένει διαθέσιμος. Η πλήρης έκδοση `three_body.ipynb` εξηγεί αναλυτικά τους υπολογισμούς. Τα δύο notebooks είναι αυτόνομα και δεν χρειάζονται άλλα αρχεία δίπλα τους.
'''
    prep = nbformat.v4.new_code_cell(
        '#@title Προετοιμασία του πειράματος — δεν χρειάζεται επεξεργασία\n\n'
        + '\n\n'.join(sources)
    )
    start = nbformat.v4.new_code_cell(
        '#@title Άνοιξε την εφαρμογή για μαθητές\n'
        'student_controls = show_student_experiment(auto_run=True)\n'
    )
    for cell in [prep, start]:
        cell.metadata.update({
            'cellView': 'form', 'collapsed': True,
            'jupyter': {'source_hidden': True},
        })
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_markdown_cell(intro), prep,
               nbformat.v4.new_markdown_cell('## Η εφαρμογή σου\n\nΤα αποτελέσματα εμφανίζονται εδώ μετά την εκτέλεση.'), start],
        metadata={
            'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
            'language_info': {'name': 'python'},
            'colab': {'name': 'three_body_students.ipynb', 'provenance': []},
        },
    )
    for index, cell in enumerate(notebook.cells):
        cell.id = hashlib.sha256((str(index) + cell.source).encode()).hexdigest()[:12]
    nbformat.validate(notebook)
    nbformat.write(notebook, ROOT / 'three_body_students.ipynb')
    print('Student edition: 2 collapsed code cells; advanced notebook unchanged.')


if __name__ == '__main__':
    build()
