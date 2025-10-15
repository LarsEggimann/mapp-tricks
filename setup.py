"""
Setup configuration for mapp_tricks package.
"""


from setuptools import setup, find_packages
from pathlib import Path

setup(
    name="mapp_tricks",
    version="0.1.9",
    author="Lars Eggimann",
    author_email="lars.eggimann@gmail.com",
    description="Reusable code developed during my PhD in the Medical Applications of Particle Physics (MAPP) group at the University of Bern.",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    include_package_data=True,
    zip_safe=False,
    packages=find_packages(),
)
