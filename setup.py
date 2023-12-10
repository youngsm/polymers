from setuptools import setup

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
    scripts=['bin/generate_dimers.py', 'bin/crtitical_exp.py'],
    install_requires=[
        'numpy',
        'numba',
        'matplotlib',
        'tqdm',
    ],
)