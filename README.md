# PEPstrMOD2

A computational framework for predicting and refining the **three-dimensional structures of peptides containing chemically modified amino acids** using state-of-the-art structure prediction methods and molecular dynamics simulations.

---

## 📌 Introduction
**PEPstrMOD2** is developed to help researchers predict the structures of peptides containing both natural and non-natural amino acids. It integrates modern protein structure prediction methods such as ESMFold and AlphaFold2 with molecular dynamics refinement using AmberTools25 and pmemd24. It represents a significant enhancement over the original PEPstrMOD framework, addressing previous limitations in scalability and flexibility.


📖 Please cite relevant content for complete details, including the algorithm behind the approach.

---

## 📚 Reference
**Jain et al.** _PEPstrMOD2:Structure Prediction of Chemically Modified Peptides_ **#Coming Soon#**

---
### 🖼️ PEPstrMOD2 Workflow Representation
![PEPstrMOD2 Workflow](https://raw.githubusercontent.com/saloni21098/PEPstrMOD2/main/images/PEPstrMOD2_workflow.png)


## 🧪 Quick Start for Reproducibility

Follow these steps to replicate the core results of our paper:

```bash
# 1. Clone the repository
git clone https://github.com/raghavagps/PEPstrMOD2.git
cd PEPstrMOD2

# 2. Set up the environment (conda recommended)
conda env create -f environment.yml
conda activate AmberTools25

# 3. Download additional tools from the Release assets
# Release page: https://github.com/raghavagps/PEPstrMOD2/releases/tag/v2.0
# Download: ambertools25.zip, pmemd24.zip, colabfold_params.zip

# 4. Extract the downloaded files in the repository root 
unzip ambertools25.zip 
unzip pmemd24.zip 
unzip colabfold_params.zip

# 5. Verify that the directory structure is: 
# PEPstrMOD2/ 
# ├── src/ 
# ├── example/
# ├── images/
# ├── list_of_modified_residues
# ├── ambertools25/ 
# ├── pmemd24/ 
# ├── colabfold_params/ 
# ├── LICENSE
# ├── environment.yml
# ├── manual.pdb
# ├── requirements.txt
# └── requirements_macos.txt

# 6. Make your working directory as src
cd src

# 7. See the available optiopns
python pepstrmod2.py -h     # for linux/windows
python pepstrmod2_macos.py -h     # for macos

# 5. Run prediction on sample input
python pepstrmod2.py -i example/test.txt -o output -m alphafold2 -e phil -s 100 -t yes -g yes -c yes 
```

## 🛠️ Installation Options

### 🔹 Standalone Installation
PEPstrMOD2 is written in **Python 3** and requires the following dependencies:

#### ✅ Required Libraries and Packages for Linux/Windows
```bash
python=3.12.2
numpy>=1.21.6,<2.0.0
matplotlib==3.10.3
requests==2.32.4
rdkit==2025.3.3
Bio==1.8.0
openpyxl==3.1.5
colabfold
colabfold[alphafold]==1.5.5
jax==0.4.23
jaxlib==0.4.23
```
#### ✅ Required Libraries and Packages for Macos
```bash
numpy>=1.21.6,<2.0.0
matplotlib==3.10.3
requests==2.32.4
rdkit==2025.3.3
Bio==1.8.0
openpyxl==3.1.5
colabfold

```

### 🔹 Installation using environment.yml
1. Create a new Conda environment:
```bash
conda env create -f environment.yml
```
2. Activate the environment:
```bash
conda activate AmberTools25
```

---

## ⚠️ Important Note
- Due to the large size of ambertools25, pmemd24 and colabfold_params, the directories have been compressed and uploaded in the Release assets of teh repository.
- Download the **zip files** from [Release Page](https://github.com/raghavagps/PEPstrMOD2/releases/tag/v2.0) 
- **Extract the files** before using the code.
- Place the extracted files inside the PEPstrMOD2 directory.

---

## 🚀 Usage

### 🔹 Minimum Usage
```bash
python pepstrmod2.py -h
```
To run an example:
```bash
python pepstrmod2.py -i example/test.txt
```

### 🔹 Full Usage
```bash
usage: python pepstrmod2.py [-h]
                   [-i INPUT]
                   [-o OUTPUT]
                   [-p PREFIX]
                   [-m METHOD {"esmfold", "alphafold2"}]
                   [-e ENVIRONMENT {"vac", "phil", "phob"}]
                   [-s SIMULATION TIME]
                   [-t TRAJECTORY {"yes", "no"}]
                   [-c CLUSTER ANALYSIS {"yes", "no"}]
                   [-g GRAPH {"yes", "no"}]
```
#### Required Arguments
| Argument                  | Description                                                                                          |
| ------------------------- | ---------------------------------------------------------------------------------------------------- |
| `-i INPUT`                | Input peptide sequence file (FASTA-like format containing peptide identifiers and sequences)         |
| `-p PREFIX`               | Prefix used for naming output files and folders (default: `test`)                                    |
| `-o OUTDIR`               | Output directory where all generated files and results will be stored (default: `output`)            |
| `-m {esmfold,alphafold2}` | Structure prediction method: `esmfold` or `alphafold2` (default: `alphafold2`)                       |
| `-e {vac,phob,phil}`      | Simulation environment: `vac` = Vacuum, `phob` = Hydrophobic, `phil` = Hydrophilic (default: `phil`) |
| `-s SIM`                  | Simulation time in picoseconds (default: `100.0`)                                                    |
| `-t {yes,no}`             | Save molecular dynamics trajectory files: `yes` or `no` (default: `no`)                              |
| `-c {yes,no}`             | Perform clustering of generated structures: `yes` or `no` (default: `no`)                            |
| `-g {yes,no}`             | Generate graphs and plots from the simulation results: `yes` or `no` (default: `no`)                 |

---

## 📂 Input & Output Files

### ✅ Input File Format

PEPstrMOD2 supports the following input formats:

1. **Single MAP Sequence (quoted string)**

```text id="d8p1kx"
"GA{d}CDEFGH"
```

2. **Text or FASTA File Containing Multiple Sequences**

```text id="t5r9vm"
>Seq1
GA{d}CDEFGH
>Seq2
ACD{ptm:Beta}FGHIK
```

Modified residues should be provided using MAP notation inside curly braces.

### ✅ Output Files

PEPstrMOD2 generates the following output files inside the specified output directory:

* Predicted peptide structures by DL models in PDB format
* Energy-minimized structures
* Final simulated structures
* Optional trajectory, clustering, rmsd and energy graph files (depending on selected arguments)

Example output structure:

```
example_test/
├── before_simulation.zip
├── after_minimization.zip
├── after_simulation.zip
├── before_simulation  #Initial Structures
    ├── Seq1_tleap.pdb
    ├── Seq2_tleap.pdb
├── after_minimization #Minimized Structures
    ├── Seq1_min.pdb
    ├── Seq2_min.pdb
├── after_simulation   #Final Structures
    ├── Seq1_final.pdb
    ├── Seq2_final.pdb
└── Seq1/
    ├── Seq1_pred.pdb
    ├── Seq1_clean.pdb
    ├── Seq1_modified.pdb
    ├── Seq1_tleap.pdb
    ....................
    ....................    
    └── Seq1_final.pdb
└── Seq2/
    ├── Seq2_pred.pdb
    ├── Seq2_clean.pdb
    ├── Seq2_modified.pdb
    ├── Seq2_tleap.pdb
    ....................
    ....................    
    └── Seq2_final.pdb
```

If no output directory is specified, all results are stored in the default folder:

```
output/
```

---

🚀 **Start predicting chemically modified peptides structures with PEPstrMOD2 today!**

🔗 Also available in Docker: [pepstrmod2](https://hub.docker.com/repository/docker/salonirara/pepstrmod2/general)
