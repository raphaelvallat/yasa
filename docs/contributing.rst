.. _Contribute:

Contribute to YASA
******************

There are many ways to contribute to YASA: reporting bugs, adding new functions, improving the documentation, etc...

Code guidelines
---------------

Before starting new code, we highly recommend opening an issue on `GitHub <https://github.com/raphaelvallat/yasa>`_ to discuss potential changes.

* YASA uses `ruff <https://github.com/astral-sh/ruff>`_ for linting and code formatting, enforced via a `pre-commit <https://pre-commit.com/>`_ hook. To install it with `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: bash

    $ uvx pre-commit install

Once installed, the hook will automatically run ``ruff`` (linting with auto-fix) and ``ruff-format`` on every commit. You can also run it manually at any time with:

.. code-block:: bash

    $ uvx pre-commit run --all-files

* Use `NumPy style <https://numpydoc.readthedocs.io/en/latest/format.html>`_ for docstrings. Follow existing examples for simplest guidance.

* When adding new functions, make sure that they are **generalizable to various situations**.

* Changes must be accompanied by **updated documentation** and examples.

* After making changes, **ensure all tests pass**. This can be done by running:

.. code-block:: bash

    $ pytest

.. _ai-assisted-prs:

AI-assisted pull requests
-------------------------

YASA has only one maintainer, and the number of AI-assisted pull requests has grown faster than they can be reviewed. AI tools are welcome. But a PR that takes a minute to generate can take hours to review, so contributors are asked to do more of that work before asking for a review. **PRs that don't follow these guidelines may be closed without review.** See also `issue #280 <https://github.com/raphaelvallat/yasa/issues/280>`_.

**Before opening a PR**

* **Open an issue first (preferred).** Describe the problem and the change you're proposing, so that we can agree it's worth doing before anyone writes or reviews code. This matters most for new features and large refactors.
* **Check other open PRs.** Look for PRs that touch the same files or functions, to avoid duplicate work and merge conflicts.
* **Base your branch on the latest** ``master``. Rebase if ``master`` moves while your PR is open.

**Opening the PR**

#. Open it as a **draft**.
#. **Run an AI review and fix the issues it finds.** If you use Claude Code, run ``/review``; with other tools, use their equivalent. Then post a summary of the review as a comment: what it found, what you fixed, and anything you chose not to fix and why.
#. **Run** ``/simplify`` (preferred, Claude Code). This helps keep the implementation small and efficient.
#. Mark the PR as ready for review only after these steps are done.

**PR description**

* **Start with a summary written by you, not by the AI**: a short paragraph explaining in your own words what the PR does and why.
* Keep it concise and in plain English. If you are fixing a bug, add the exact code to reproduce the bug and show the output before and after the fix.
* **Report the exact model you used**, e.g. *"Claude Opus 5.5 via Claude Code"*.

**Scope and size**

* **Keep PRs small and focused.** A PR should do one thing. If your change is large, split it into a series of small PRs stacked on top of each other.
* **Add tests only when they're needed.** AI agents tend to write too many tests, which slows down the test suite and makes it harder to maintain. Add a test only if it covers new behavior, a bug fix (a regression test), or a critical edge case. Remove tests that duplicate existing ones or only check trivial things.

The code guidelines above still apply, and you are responsible for every line you submit. If you can't explain a change, please don't include it.

Checking and building documentation
-----------------------------------

YASA's documentation (including docstring in code) uses ReStructuredText format,
see `Sphinx documentation <https://www.sphinx-doc.org/en/master/>`_ to learn more about editing them. The code
follows the `NumPy docstring standard <https://numpydoc.readthedocs.io/en/latest/format.html>`_.

All changes to the codebase must be properly documented. To ensure that documentation is rendered correctly, the best bet is to follow the existing examples for function docstrings. If you want to test the documentation locally, you will need to install the required packages:

.. code-block:: bash

    $ uv pip install --group=docs --editable .

and then from the root of the repository do:

.. code-block:: bash

    $ make -C docs html
