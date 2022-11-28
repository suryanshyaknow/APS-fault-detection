from setuptools import find_packages, setup
from typing import List


REQUIREMENTS_FILE = "requirements.txt"
HYPHEN_E_DOT = "-e ."


def get_requirements() -> List[str]:
    with open(REQUIREMENTS_FILE) as f:
        requirements = f.readlines()

    requirements = [i.replace('\n', '') for i in requirements]
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
    return requirements


setup(
    name='src',
    version="0.0.1",
    author="suryanshyaknow",
    author_email="suryanshgrover1999@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)
