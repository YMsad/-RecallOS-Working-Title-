"""RecallOS - packaging for local development.

Supports editable installs:

    pip install -e .

Package name:   recallos-socratic
Console script: recallos   (runs ``cli.main``)
"""

from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup

HERE = Path(__file__).resolve().parent
README = (HERE / "README.md").read_text(encoding="utf-8")

setup(
    name="recallos-socratic",
    version="0.1.0",
    description="RecallOS - 基于苏格拉底追问法的 AI 学习伴侣",
    long_description=README,
    long_description_content_type="text/markdown",
    author="YMsad",
    license="MIT",
    url="https://github.com/YMsad/-RecallOS-Working-Title-",
    python_requires=">=3.10",
    packages=find_packages(exclude=("tests", "tests.*")),
    include_package_data=True,
    install_requires=[
        "pydantic==2.13.4",
        "pydantic-settings==2.14.2",
        "python-dotenv==1.2.2",
        "httpx==0.28.1",
        "streamlit==1.60.0",
    ],
    entry_points={
        "console_scripts": [
            "recallos=cli.main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)