from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        name="a_star",
        sources=["a_star.pyx"],
        include_dirs=[numpy.get_include()]  # Esto asegura que numpy esté incluido correctamente
    )
]

setup(
    ext_modules=cythonize(extensions)
)