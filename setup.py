"""
Setup script for the RAG evaluation framework package.
"""
from setuptools import setup, find_packages

setup(
    name="rag_evaluation",
    version="0.1.0",
    description="A framework for evaluating different RAG approaches",
    author="Master Thesis Author",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        # Core dependencies are managed through requirements.txt
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.1",
        ],
    },
) 