# Changes — 2026-10-24

## Clean, readable GitHub package

- Readable standalone notebook: 40 cells, including 18 ordinary Python code cells.
- Consistently formatted Python, HTML and JavaScript sources.
- One interactive menu and one Canvas player; removed the redundant legacy menu/player from `presets.py`.
- Preserved the numerical model and the legacy numerical CLI entry points.
- Deterministic notebook generation and cell IDs.
- Widget tests share an ordinary function instead of parsing and executing fragments of another test script.
- Tests locate files from the repository layout, without assuming an outer `software` folder.
- Numerical validation generates all required browser fixtures.
- Updated README, student guide, numerical report, dependency files and developer commands.
- Re-ran 19 numerical configurations / 108 integrations / 548 assertions, standalone notebook execution and 26 Chromium player checks.

No live Colab or Binder frontend session was tested. Python widget callbacks and the standalone HTML player were tested separately. These checks cover the listed cases and are not a proof of chaos or a universal accuracy guarantee.
