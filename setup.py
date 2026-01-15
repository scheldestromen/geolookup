from setuptools import setup, find_packages

setup(
    name="dijkzicht",
    version="0.1.0",
    description="Dijkzicht je Dijk - Dike profile analysis and visualization",
    author="scheldestromen",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "plotly"
    ],
    python_requires='>=3.7',
)
