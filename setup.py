from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='deeptransyt',
    version='0.0.12-py38',
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
        "biopython==1.83",
        "fair_esm==2.0.0",
        "numpy==1.24.4",
        "pandas==2.0.3",
        "pytorch_lightning==2.4.0",
#        "tensorflow==2.17.0",
        "torch==2.4.1",
    ],
    entry_points={
        'console_scripts': [
            'run-predictions=deeptransyt.main:main',
        ],
    },
)