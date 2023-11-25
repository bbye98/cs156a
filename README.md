# CS/CNS/EE 156a: Learning Systems (Fall 2023)

A repository of my solutions to the homework assignments, bonus 
exercise, and final exam for the Fall 2023 installment of CS/CNS/EE 
156a: Learning Systems taught by Professor Yaser Abu-Mostafa at the
California Institute of Technology.

## Pre-requisites

Requires Python 3.10 and the following packages:

* Matplotlib
* NumPy
* pandas
* requests
* scikit-learn
* SciPy

## Repository Structure

* The `cs156a.py` file in the main directory acts as a Python module with 
  helper functions that are used throughout the homework assignments and 
  final exam.
* The `hw` directory contains subsubdirectories for each homework 
  assignment. Within the subdirectories, the `hw*.docx` and `hw*.pdf`
  files are the writeups, the `hw*.py` and `hw*.ipynb` files are
  functionally equivalent Python scripts and Jupyter Notebooks for the
  entire assignment, and the `p**_**.py` files are Python
  scripts for individual problems.
* Similarly, the `final` directory contains the writeup in `final.docx`
  and `final.pdf`, a Python script and Jupyter Notebook for the entire
  exam in `final.py` and `final.ipynb`, and Python scripts for 
  individual problems in the `p**_**.py` files.
* The `bonus` directory contains the writeup in `bonus.docx` and 
  `bonus.pdf`, completed bonus exercise Jupyter Notebook templates in
  `Bonus_Part_1.ipynb` and `Bonus_Part_2.ipynb`, and model data and 
  figures generated during training in the various subdirectories,
  `*.npy`, and `*.png` files.

Note that the `hw*.py` and `hw*.ipynb` files, `final.py`, and 
`final.ipynb` import `cs156a.py`, while the `p**_**.py` files are 
standalone Python scripts that only depend on the Python standard 
library, Matplotlib, NumPy, pandas, requests, scikit-learn, and/or 
SciPy.