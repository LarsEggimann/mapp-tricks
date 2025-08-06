"""
Setup configuration for mapp_tricks package.
"""

from setuptools import setup, find_packages

setup(
    name="mapp_tricks",
    version="0.1.0",
    author="Lars Eggimann",
    author_email="lars.eggimann@gmail.com",
    description="Reusable code developed during my PhD in the Medical Applications of Particle Physics group at the University of Bern. ",
    include_package_data=True,
    zip_safe=False,
    packages=['mapp_tricks'],
)
