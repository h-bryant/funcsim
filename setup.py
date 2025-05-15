import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="funcsim",
    version="0.1.00",
    license="bsd-3-clause",
    author="Henry Bryant",
    author_email="henry@tamu.edu",
    description="functional, simple stochastic simulation",
    keywords=["functional", "stochastic", "simulation"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/h-bryant/funcsim",
    packages=setuptools.find_packages(),
    install_requires=[
        'copulae>=0.8.0',
        'numpy>=2.2.5',
        'pandas>=2.2.3',
        'plotly>=6.0.1',
        'scipy>=1.15.3',
        'xarray>=2025.4.0'
    ],
    test_requires=[
        'pytest',
        'sphinx'
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis"
    ],
)
