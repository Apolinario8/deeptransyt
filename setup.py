from setuptools import setup, find_packages
import re
import os

def get_version():
    version_file = os.path.join(os.path.dirname(__file__), "src", "deeptransyt", "__init__.py")
    with open(version_file, "r") as f:
        match = re.search(r'__version__ = "(.*?)"', f.read())
        return match.group(1) if match else "0.0.0"
    
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='deeptransyt',
    version=get_version(),
    license="MIT",
    python_requires='>=3.8',
    description="Transporters annotation using LLM's",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Gonçalo Apolinário Cardoso',
    author_email='goncalocardoso2016@gmail.com',
    url='https://github.com/Apolinario8/deeptransyt',  
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[ 
        "Bio==1.6.2",
        "numpy==1.24.4",
        "pandas==2.0.3",
        "pytorch_lightning==2.4.0",
        "torch==2.4.1",
        "transformers==4.46.3"
    ],
    entry_points={
        'console_scripts': [
            'deeptransyt=deeptransyt.main:cli_main',
        ],
    },
)