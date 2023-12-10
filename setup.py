from setuptools import setup
from glob import glob
import os

scripts=glob(os.path.join(os.path.dirname(__file__), 'bin/*.py'))
setup(
    name="polymers",
    version="0.1",
    author="Sam Young",
    author_email="youngsam@stanford.edu",
    include_package_data=False,
    description='',
    license='MIT',
    keywords='polymers',
    packages=['polymers'],
    scripts=scripts,
    install_requires=[
        'numpy',
        'numba',
        'matplotlib',
        'tqdm',
    ],
)