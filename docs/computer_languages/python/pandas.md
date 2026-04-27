# Pandas

Pandas is a Python package providing fast, flexible, and expressive data structures designed to make working with relational or labeled data easy and intuitive. It aims to be the fundamental high-level building block for practical, real-world data analysis in Python.

[Pandas](https://pandas.pydata.org) is built on top of [NumPy](https://numpy.org/) and integrates well with the broader scientific Python ecosystem. 

TODO: Understand this. [BEGIN]
It is fast, and many low-level algorithmic components are heavily optimized in [Cython](https://cython.org/) code.
[END]

The two primary pandas data structures are:

- [`Series`](https://pandas.pydata.org/docs/reference/api/pandas.Series.html#pandas.Series) (1-dimensional)
- [`DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html#pandas.DataFrame) (2-dimensional)

These handle most common use cases in finance, statistics, social science, and engineering.

All pandas data structures are value-mutable (contained values can change) but not always size-mutable. For example, a `Series` length cannot be changed, while columns can be inserted into a `DataFrame`. Most methods return new objects and leave input data unchanged.

NumPy arrays have one dtype for the entire array, while pandas `DataFrame` objects can have one dtype per column.

For production code, prefer optimized pandas accessors over ad hoc indexing patterns:

- [`DataFrame.at()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.at.html#pandas.DataFrame.at)
- [`DataFrame.iat()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iat.html#pandas.DataFrame.iat)
- [`DataFrame.loc()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.loc.html#pandas.DataFrame.loc)
- [`DataFrame.iloc()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html#pandas.DataFrame.iloc)

## Resources

Useful quick reference: [Pandas User Guide (10 minutes to pandas)](https://pandas.pydata.org/docs/user_guide/10min.html#minutes-to-pandas)

Example notebook: [pandas_basics.ipynb](https://github.com/adi14041999/learn_python/blob/main/pandas_basics.ipynb)

Source repository: [adi14041999/learn_python](https://github.com/adi14041999/learn_python)