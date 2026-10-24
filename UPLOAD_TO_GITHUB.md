# Upload this complete update

1. Extract the ZIP.
2. Open https://github.com/gkalemis/three-body-experiment/upload/main
3. Upload the **contents** of the extracted folder to the repository root, including `tests` and `results/interactive_validation`. Do not add an extra enclosing `software` or `github_package` folder.
4. Replace files with the same paths. Existing historical results can remain; this update does not require deleting them.
5. Commit the update, then reopen https://colab.research.google.com/github/gkalemis/three-body-experiment/blob/main/three_body.ipynb and run all cells.

The ZIP contains the current source, notebook, three main documentation files, dependency manifests, tests and their new result data. Generated personal runs in `results/interactive` are excluded. Browser screenshots, cache folders and dependencies are excluded.

For a smaller web upload, the numerical result folders are optional for ordinary use; source tests can reproduce them. Keep the source, notebook, documentation and dependency manifests together.
