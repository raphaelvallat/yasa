.. _tutorials:

Tutorials
#########

Step-by-step guides for common YASA workflows.

.. toctree::
    :hidden:

    quickstart
    hypnogram
    migrate

.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card:: Quickstart
        :link: quickstart
        :link-type: doc

        A walkthrough of YASA's core features using real PSG data: loading
        recordings, working with hypnograms, spectral analysis, and event detection.

    .. grid-item-card:: Working with Hypnograms
        :link: hypnogram
        :link-type: doc

        Learn how to create, manipulate, analyze, and visualize sleep hypnograms
        using the :py:class:`~yasa.Hypnogram` class introduced in YASA 0.7.

    .. grid-item-card:: Migrating to the Hypnogram class
        :link: migrate
        :link-type: doc

        Coming from YASA 0.6? Step-by-step patterns for updating code that used
        integer arrays and the old standalone functions to the new
        :py:class:`~yasa.Hypnogram` API.
