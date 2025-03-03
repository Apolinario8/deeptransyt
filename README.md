# deeptransyt

![Downloads per month](https://pepy.tech/badge/deeptransyt/month)
![License](https://img.shields.io/pypi/l/deeptransyt.svg)
![Wheel Support](https://img.shields.io/pypi/wheel/deeptransyt.svg)
![Python Versions](https://img.shields.io/pypi/pyversions/deeptransyt.svg)

## 🚀 Installation  

### **0. Create conda environment **
```bash
conda create -n deeptransyt python=3.x # or python=3.9, python=3.10, python=3.11
conda activate deeptransyt
```

### **1. Install via `pip`**
```bash
pip install deeptransyt
```

### **2. Cloning the repository**
```bash
git clone https://github.com/Apolinario8/deeptransyt.git
cd deeptransyt
pip install .
```

---

## Dataset and Models

The dataset and trained models used for this project are available on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14946179.svg)](https://doi.org/10.5281/zenodo.14946179)


<!--## API Reference and User Guide available on [Read the Docs](https://deeptransyt.readthedocs.io/) -->

<!-- [![Read the Docs](https://raw.githubusercontent.com/psf/requests/main/ext/ss.png)](https://requests.readthedocs.io) -->

## Usage

DeepTransyt is a command-line tool designed for predicting transporter proteins from sequence data. You can run it with different configurations depending on your input data.  

### **Basic Command**  
To run DeepTransyt, execute the following command:  

```bash
python deeptransyt.py --input_file <your_genome.faa> --output_dir <output_directory> 
```
This will run the pipeline and save the results in the specified output directory.

Required Arguments
--output_dir: Specifies the directory where the results will be stored.
Optional Arguments
--input_file <path>: Path to a FASTA file containing genome sequences.
--gpu <index>: Specify the GPU index to use (default: 2).
--embeddings_file <path>: Path to an existing embeddings file to skip feature extraction.
--binary_threshold <value>: Sets the threshold for binary classification (default: 0.5).

Example Commands
Running with a genome FASTA file
```bash
python deeptransyt.py --input_file <genome.faa> --output_dir <results/> 
```

Using precomputed embeddings
```bash
python deeptransyt.py --embeddings_file embeddings.npy --output_dir results/
```

It's also possible to run in a jupyter notebook. An example is provided in [notebook.ipynb](https://github.com/Apolinario8/deeptransyt/blob/main/notebook.ipynb)

## 📜 License
This project is licensed under the MIT License. See the `LICENSE` file for details.


## DeepTransyt's Workflow
![alt text](https://github.com/Apolinario8/deeptransyt/blob/main/annotation_tool_workflow.png?raw=true)



