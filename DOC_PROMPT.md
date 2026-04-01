# Documentation System Setup Prompt

Use this prompt to replicate the documentation system used in this project for other Python libraries.

---

**Task:** Set up a professional documentation system using Sphinx, Furo theme, and Jupyter Notebook integration.

**Context:** The project is a Python library that needs high-quality documentation including API reference, user guide, and executable examples from Jupyter notebooks.

**Requirements:**

1.  **Dependencies:**
    - Add a `docs` dependency group in `pyproject.toml` (or `requirements-docs.txt`) with the following packages:
        ```python
        sphinx>=8.0
        nbsphinx>=0.9.7
        nbsphinx-link
        furo
        ipykernel
        sphinx-rtd-theme>=3.0.2  # (Optional fallback or specific components)
        sphinxcontrib-bibtex>=2.6.5
        ```

2.  **Configuration (`conf.py`):**
    - Set `html_theme = "furo"`.
    - enable extensions:
        ```python
        extensions = [
            "nbsphinx",
            "nbsphinx_link",
            "sphinx.ext.autodoc",
            "sphinx.ext.autosummary",
            "sphinx.ext.intersphinx",
            "sphinx.ext.mathjax",
            "sphinx.ext.napoleon",
            "sphinx.ext.viewcode",
            "sphinxcontrib.bibtex",
        ]
        ```
    - Configure `nbsphinx` to not execute notebooks during build (unless desired):
        ```python
        nbsphinx_execute = 'never'
        nbsphinx_allow_errors = True
        ```
    - Configure `bibtex` if references are needed.

3.  **Folder Structure:**
    - `docs/`: Root documentation folder.
    - `docs/examples/`: Place `.nblink` files here to link to notebooks in the project root or examples directory.
    - `docs/_static/`: Custom CSS/JS.
    - `docs/api/`: API reference stubs.
    - `docs/user_guide/`: Narrative documentation.

4.  **Notebook Integration:**
    - Use `nbsphinx-link` to include notebooks from outside the `docs` directory without copying them. Create `.nblink` files in `docs/examples/` like:
        ```json
        { "path": "../../examples/my_notebook.ipynb" }
        ```

5.  **Build Automation:**
    - Ensure `make html` (via Makefile) works effectively.
    - Use `uv`, run with `uv run --group docs make -C docs html`.

**Output:**

- A fully functional `docs/` directory with configured `conf.py`.
- Updated `pyproject.toml` with necessary dependencies.
- Successful build of HTML documentation.
