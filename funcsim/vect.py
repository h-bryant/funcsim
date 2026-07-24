"""
Vectorize() with support for decorating methods; for example::

    from scipy.stats import rv_continuous
    from scipy_ext import vectorize

    class dist(rv_continuous):
        @vectorize(excluded=('n',), otypes=(float,))
        def _cdf(self, x, n):
            if n < 5:
                return f(x)  # One expensive calculation.
            else:
                return g(x)  # A different expensive calculation.

Original Work (scikit-gof) Copyright (c) 2015 Wojciech Ruszczewski <scipy@wr.waw.pl>

Modified Work Copyright (c) 2020 h-bryant
"""
import copy as _copy
from collections.abc import Iterable
# from __future__ import division

from numpy import arange, stack, vectorize as numpy_vectorize


class _vectorize(numpy_vectorize):
    """
    Method decorator, working just like `numpy.vectorize()`.
    """
    def __get__(self, instance, owner):
        # Vectorize stores the decorated function (former "unbound method")
        # as pyfunc.  Return a fresh copy bound to this instance: mutating
        # self.pyfunc in place would permanently capture the first instance
        # that ever accessed the method, silently redirecting calls from
        # every later instance to the first one's state.
        if instance is None:
            return self
        bound = _copy.copy(self)
        bound.pyfunc = self.pyfunc.__get__(instance, owner)
        return bound


def vectorize(*args, **kwargs):
    """
    Allows using `@vectorize` as well as `@vectorize()`.
    """
    if args and callable(args[0]):
        # Guessing the argument is the method.
        return _vectorize(args[0])
    else:
        # Wait for the second call.
        return lambda m: _vectorize(m, *args, **kwargs)


def varange(starts, count):
    """
    Vectorized `arange()` taking a sequence of starts and a count of elements.

    For example::

        >>> varange(1, 5)
        array([1, 2, 3, 4, 5])

        >>> varange((1, 3), 5)
        array([[1, 2, 3, 4, 5],
               [3, 4, 5, 6, 7]])
    """
    if isinstance(starts, Iterable):
        return stack([arange(s, s + count) for s in starts])
    # scalar start: return 1-D, as the docstring and doctest promise
    return arange(starts, starts + count)
