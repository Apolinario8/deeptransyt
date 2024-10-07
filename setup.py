from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='deeptransyt',
    version='0.0.11',
    description="Transporters annotation using LLM's",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Gonçalo Apolinário Cardoso',
    author_email='goncalocardoso2016@gmail.com',
    url='https://github.com/Apolinario8/deeptransyt',  
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[ 
        "Bio==1.7.1",
        "biopython==1.84",
        "fair_esm==2.0.0",
        "numpy==1.26.4",
        "pandas==2.2.2",
        "pytorch_lightning==2.3.3",
        "tensorflow==2.17.0",
        "torch==2.3.0",
    ],
    entry_points={
        'console_scripts': [
            'run-predictions=deeptransyt.main:main',
        ],
    },
)