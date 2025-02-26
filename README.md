# deeptransyt

## 🚀 Installation  

You can install `deeptransyt` in two ways:

### **1. Install via `pip`**
```bash
pip install deeptransyt
```

### **2. Install from GitHub**
If you want the latest development version:
```bash
git clone https://github.com/yourusername/deeptransyt.git
cd deeptransyt
pip install .
```

---

## ⚡ Usage  

### **1. Using the CLI**
Once installed, you can run `deeptransyt` from the command line:

```bash
deeptransyt --organism_id eco hsa \
            --input_dir ./data \
            --output_dir ./results \
            --gpu 0
```

For a full list of available options, run:
```bash
deeptransyt --help
```

---

### **2. Running in a Jupyter Notebook**
You can also use `deeptransyt` programmatically in a Python script or Jupyter notebook:

```python
from deeptransyt.main import main as run_deeptransyt

run_deeptransyt(
    organism_id=["eco", "hsa"],
    input_dir="./data",
    output_dir="./results",
    gpu=0
)
```

## 📜 License
This project is licensed under the MIT License. See the `LICENSE` file for details.

![alt text](https://github.com/Apolinario8/deeptransyt/blob/main/annotation_tool_workflow.png?raw=true)




