from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="omad",
    version="0.1.0",
    description="Ocean Maritime Anomaly Detection - Pipeline CLI Tool",
    author="OMAD Contributors",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "omad=omad.cli:app",
        ],
    },
    python_requires=">=3.8",
)
