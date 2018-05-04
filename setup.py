import os
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize


version = os.environ.get('PKG_VERSION', '0.0.0dev')


with open('./polarbear/__version__.py', 'w') as f:
    f.write('__version__ = "{version}"'.format(version=version))


ext_modules = [
    # Extension(
    #     "polarbear.core.arr",
    #     ["./polarbear/core/arr.pyx"],
    #     extra_compile_args=['-fopenmp'],
    #     extra_link_args=['-fopenmp'],
    # )
]

setup(
    name='Polarbear',
    version=version,
    packages=find_packages(),
    scripts=[],
    install_requires=[],
    ext_modules=cythonize(ext_modules),
    package_data={},
    author='valoox',
    url='http://github.com/valoox/polarbear',
)

