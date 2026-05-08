# Arena framework

<div align="center">
<a href="https://gitlab.kuleuven.be/u0153320/arena-framework"><img src="https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black" /></a>
<a href="https://gitlab.kuleuven.be/u0153320/arena-framework"><img src="https://img.shields.io/badge/Windows-0078D6?st&logo=windows&logoColor=white" /></a>
</div>

## Description

A Python framework for **A**gile and **Re**liable **Na**vigation of autonomous guided vehicles (AGV).

**Authors:** [Sonia De Santis](https://www.mech.kuleuven.be/en/pma/research/meco/people/00153320) and [Alejandro Astudillo](https://scholar.google.com/citations?user=9ONkJZAAAAAJ).



## Installation

### Option 1: Installing with pip
You can install this package (ideally into a virtual environment) via pip using the following command:

```
pip install git+https://gitlab.kuleuven.be/u0153320/arena-framework.git@main
```

### Option 2: Installing from cloned repository
Alternatively, you can clone this repository and install the package from source. You just need to (i) clone the repository, (ii) move into Arena's root directory, and (iii) run the `setup.py` script with the `install` option. It will install your application into the virtualenv site-packages folder and also download and install all dependencies:

```
git clone https://gitlab.kuleuven.be/u0153320/arena-framework.git
cd arena-framework
python setup.py install 
```
You could also use the `develop` option, instead of `install`, during the execution of `setup.py` as `python setup.py develop`. 
This has the advantage of just installing a link to the site-packages folder instead of copying the data over. You can then modify/update the source code without having to run `python setup.py install` again after every change.

Another option is to run `pip install -e .` from Arena's root directory, instead of running `python setup.py develop`.


## Submitting an issue

Please submit an issue if you want to report a bug or propose new features.
