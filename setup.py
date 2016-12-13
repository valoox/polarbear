import os
from setuptools import setup, find_packages

version = os.environ.get('PKG_VERSION', '0.0.0dev')

with open('./polarbear/__version__.py', 'w') as f:
    f.write('__version__ = "{version}"'.format(version=version))

setup(
    name='Polar Bear',
    version=version,
    packages=find_packages(),
    scripts=[],
    install_requires=[],
    package_data={},
    author='valoox',
    url='http://github.com/valoox/polarbear',
)

