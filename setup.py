from setuptools import setup

requirements = ['numpy', 'matplotlib', 'pandas', 'numba', 'iqrm']
packages = ['caspy_search']

setup(name = "caspy_search",
        version='1.0',
        description = "Package to search filterbank data for single pulses using FDMT and boxcar",
        author = "Vivek Gupta",
        author_email = "vivek.gupta@csiro.au",
        install_requires = requirements,
        python_requires = '>3.6',
        packages=packages,
        dependency_links = ['https://github.com/FRBs/sigpyproc3/tarball/main'],
        entry_points = {'console_scripts': ['search_cas_fil=caspy_search.search_cas_fil:main']}
        )