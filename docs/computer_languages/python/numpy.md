# NumPy

NumPy is the fundamental package for scientific computing in Python. It provides a multidimensional array object, derived objects (such as masked arrays and matrices), and a wide range of fast operations on arrays, including:

- mathematical operations
- logical operations
- shape manipulation
- sorting and selection
- I/O
- discrete Fourier transforms
- basic linear algebra
- basic statistics
- random simulation

TODO: Understand this. [BEGIN]
When using CPython, interpretation overhead is usually small. During computationally intense NumPy operations, Python calls optimized compiled binaries (typically C/C++), which is one reason NumPy is fast.
[END]

At the core of NumPy is the `ndarray` object, which encapsulates homogeneous n-dimensional arrays.

## NumPy arrays vs Python lists

- **Fixed size:** NumPy arrays are fixed-size at creation (resizing creates a new array), while Python lists can grow dynamically.
- **Homogeneous dtype:** NumPy array elements generally share one data type and memory size (object arrays are an exception).
- **Efficient numerical ops:** NumPy supports advanced operations over large datasets with less code and better performance than built-in Python sequences.

Under the hood, a Python list stores pointers to Python objects. For a list of integers, each integer is a full Python object with per-object metadata (type, refcount, etc.). Accessing elements involves pointer dereferencing and type handling overhead.

By contrast, a NumPy array stores data in a contiguous memory block with shared metadata (like dtype and shape). This improves cache locality and reduces per-element overhead, especially in repeated numerical operations. For example, element type is known in advance from array metadata, so operations avoid repeated per-element type checks, which speeds up computation.

## Vectorization

TODO: Understand this. [BEGIN]
Vectorization means writing operations without explicit Python loops/indexing for element-wise work; the loops run behind the scenes in optimized compiled code.

### `for` loop vs vectorization: which is faster?

In most numerical workloads, **vectorized NumPy operations are much faster** than Python `for` loops.

Why:

- A Python `for` loop executes in the Python interpreter, with per-iteration overhead (bytecode execution, object handling, type checks).
- Vectorized NumPy operations move that loop into optimized compiled code (C/C++/SIMD-enabled routines), reducing Python-level overhead.
- NumPy arrays use contiguous memory, which improves cache efficiency during bulk operations.

As a result, vectorization can be many times faster (often one or more orders of magnitude) for large arrays.

### Important caveats

- For very small arrays, speed differences may be minor.
- Chaining many vectorized expressions can create temporary arrays, increasing memory usage and sometimes reducing speed.
[END]

In practice, prefer vectorization for array math, then profile if performance is critical.

### Advantages

- more concise and readable code
- fewer lines of code and often fewer bugs
- code that more closely matches mathematical notation

## Broadcasting

Broadcasting is NumPy's implicit element-by-element behavior across compatible shapes. It applies not only to arithmetic, but also to logical, bitwise, and many functional operations.

For formal broadcasting rules, see [NumPy Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html#basics-broadcasting).

## Resources

[Crash course](https://www.youtube.com/watch?v=9JUAPgtkKpI)

[Google's NumPy UltraQuick Tutorial](https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/numpy_ultraquick_tutorial.ipynb)

Example notebook: [numpy_basics.ipynb](https://github.com/adi14041999/learn_python/blob/main/numpy_basics.ipynb)

Source repository: [adi14041999/learn_python](https://github.com/adi14041999/learn_python)