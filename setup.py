from setuptools import setup, find_packages

setup(
    name="gor-toolkit",
    version="0.1.1",
    description="Measurement toolkit for two-layer meta-documents (Geometry of Reporting)",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "networkx>=2.6.0",
        "scikit-learn>=1.0.0",
        "pyyaml>=5.4.0",
        "jsonschema>=4.0.0",
        "scipy>=1.7.0",
        "mecab-python3",
        "unidic-lite",
        "pytest"
    ],
    python_requires=">=3.9",
)
