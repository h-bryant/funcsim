import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="funcsim",
    version="0.1.5",
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
        'numpy>=2.2.0',
        'pandas>=2.2.0',
        'scipy>=1.16.0',
        'xarray>=2025.4.0'
    ],
    extras_require={
        "plotting": ["plotly>=6.0.0", "jupyter"],
        "copula": ['copulae>=0.8.0'],
        "docs": ['sphinx', 'sphinx_autodoc_typehints'],
        "full": ["plotly>=6.0.0", "copulae>=0.8.0", 'sphinx', 'sphinx_autodoc_typehints'],
    },
    test_requires=[
        'pytest',
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis"
    ],
)
