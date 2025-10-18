# Machine Learning in Physics
## Code Repository for PHY6937/PHY4936 Special Topics Course

## Introduction
This repository contains machine learning tutorials associated with the Special Topics Course on Machine Learning in Physics at Florida State University, Department of Physics. The course is designed for graduate students and senior undergrads who have little to no experience with machine learning, but have some experience with Python.


## Dependencies
The notebooks in this repository depend on one or more of several well-known
well-engineered and free Python modules.

| __modules__   | __description__     |
| :---          | :---        |
| pandas        | data table manipulation, often with data loaded from csv files |
| numpy         | array manipulation and numerical analysis   |
| scipy         | scientific computing    |
| sympy         | an excellent symbolic mathematics module |
| matplotlib    | a widely used plotting module for producing high quality plots |
| imageio      | photo-quality image display module |
| scikit-learn  | easy to use machine learning toolkit |
| pytorch       | a powerful, flexible, research-level machine learning toolkit |
| iminuit | a rewrite of the venerable CERN minimizer Minuit |
| emcee | one of many, many, Markov chain Monte Carlo modules |
| tqdm         | progress bar |
| joblib | module to save and load Python objects |
| importlib | importing and re-importing modules |

##  Installation
The simplest way to install these Python modules is first to install a software release system. 
You could just bite the bullet and install Anaconda! However, it may be better to
miniconda3, which is a slim version of Anaconda, on your laptop. Do so by following the instructions at:

https://docs.anaconda.com/miniconda/

Software release systems such as Anaconda (__conda__ for short) make
it possible to have several separate self-consistent named
*environments* on a single machine, say your laptop. For example, you
may need to use Python 3.11.8 and an associated set of compatible
packages and at other times you may need to use Python 3.12.4 with
packages that require that particular version of Python.  If you install software without using *environments* there is
the danger that the software on your laptop will eventually become
inconsistent. Anaconda (and its lightweight companion miniconda)
provide a way, for example, to create a software *environment*
consistent with Python 3.11.8 and another that is consistent with
Python 3.12.4.  For example,
one package may work only with a given version of numpy, while another
requires a different version. In principle, having different versions of numpy on
your machine, just
like having different versions of Python, is not a problem if one uses
environments.

Of course, like anything human make, miniconda3 is not
perfect. There are times when the only solution is to delete an
environment and rebuild by reinstalling the desired packages.

### Miniconda3

After installing miniconda3, It is a good idea to update conda using the command
```bash
conda update conda
```
#### Step 1 
Assuming conda is properly installed and initialized on your laptop, you can create an environment, here called *mlphysics*. 
```bash
conda create --name mlphysics
```
and activate it by doing
```bash
conda activate mlphysics
```
You need create the environment only once, but you must activate the desired environment whenever you create a new terminal window.

#### Step 2 
Install root, python, numpy, … (Tip: search the web for **conda install** and the
module name to get the exact syntax just in case it has changed.)
```
	conda install –c conda-forge root
```
If all goes well, this will install a recent version of the [ROOT](https://root.cern.ch) package from CERN as well as *Python* and several Python modules including *numpy*.

#### Step 3
Install *pytorch*, *matplotlib*, *scikit-learn*, etc.
```bash
	conda install –c conda-forge pytorch
	conda install –c conda-forge matplotlib
	conda install –c conda-forge scikit-learn
	conda install –c conda-forge pandas
	conda install –c conda-forge sympy
	conda install –c conda-forge imageio.v3
	conda install –c conda-forge jupyter
```
Again be sure to check the exact syntax. This does change from time to time!

#### Step 4
Install __git__ if it is not yet on your system, then download the **MLinPhysics** package.
```bash
	conda install –c conda-forge git
	mkdir tutorials
	cd tutorials
	git clone https://github.com/hbprosper/MLinPhysics
```
In the above the package MLinPhysics has been downloaded into a directory called *tutorials*.

#### Step 5

Open a new terminal window, navigate to the directory (folder) that contains **MLinPhysics** and run  **jupyter lab** in that window (in blocking mode).
```bash
	jupyter lab
```
If all goes well, the jupyter lab environment will appear in your default browser. 
Navigate to the MLinPhysics directory and under the *Files* menu item, click on the notebook *test.ipynb* and execute it. This notebook tries to import several Python modules. If it does so without error messages, you are ready to try out the other notebooks.

