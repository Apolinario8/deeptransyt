from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='deeptransyt',
    version='0.0.3',
    description="Transporters annotation using LLM's",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Gonçalo Apolinário Cardoso',
    author_email='goncalocardoso2016@gmail.com',
    url='https://github.com/Apolinario8/deeptransyt',  
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[ 
        "Bio",
        "biopython",
        "fair_esm==2.0.0",
        "numpy==2.0.1",
        "pandas==2.2.2",
        "pytorch_lightning==2.2.5",
        "tensorflow==2.17.0",
        "torch==2.3.0",
    ],
    entry_points={
        'console_scripts': [
            'run-predictions=deeptransyt.main:main',
        ],
    },
)