"""Setup module for building TRACE."""


from distutils.core import Extension

import numpy
from Cython.Build import cythonize
from setuptools import setup

extensions = [
    Extension(
        "trace_utils",
        ["tracehmm/trace_utils.pyx"],
        include_dirs=[numpy.get_include()],
    )
]

setup_args = dict(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "profile": False,
        },
    )
)
setup(**setup_args)
