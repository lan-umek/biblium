from setuptools import setup, find_packages
setup(
    name="biblium",
    version="2.16.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=["pandas>=1.5.0", "numpy>=1.21.0", "matplotlib>=3.5.0", "networkx>=2.6", "scikit-learn>=1.0.0"],
)
