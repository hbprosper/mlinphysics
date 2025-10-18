# Machine Learning in Physics
## Code Repository for a Special Topics Course

## Introduction
This repository contains machine learning tutorials and lectures associated with the Special Topics Course on Machine Learning in Physics at Florida State University, Department of Physics. The course is designed for graduate students and senior undergrads who have little to no experience with machine learning, but have some experience with Python.


## Dependencies
The notebooks in this repository depend on one or more of several well-known
well-engineered and free Python modules.

| __modules__   | __description__     |
| :---          | :---        |
| pytorch       | a powerful, flexible, research-level machine learning toolkit |
| scikit-learn  | easy to use machine learning toolkit |
| numpy         | array manipulation and numerical analysis   |
| matplotlib    | a widely used plotting module for producing high quality plots |
| scipy         | scientific computing    |
| sympy         | an excellent symbolic mathematics module |
| pandas        | data table manipulation, often with data loaded from csv files |
| imageio      | photo-quality image display module |
| iminuit | a rewrite of the venerable CERN minimizer Minuit |
| emcee | one of many, many, Markov chain Monte Carlo modules |
| tqdm         | progress bar |
| joblib | module to save and load Python objects |
| importlib | importing and re-importing modules |

##  Installation
The simplest way to install these Python modules is first to install a software environment system. 
You could just bite the bullet and install Anaconda! However, it may be better to install
**miniconda3**, which is a slim version of Anaconda, on your laptop. Do so by following the instructions at:

https://docs.anaconda.com/miniconda/

Software environment systems such as Anaconda (__conda__ for short) make
it possible to have several separate self-consistent named
**environments** on a single machine, say your laptop. For example, you
may need to use Python 3.11.8 and an associated set of compatible
packages and at other times you may need to use Python 3.12.4 with
packages that require that particular version of Python.  If you install software without using environments there is
the danger that the software on your laptop will eventually become
inconsistent. Anaconda (and its lightweight companion miniconda)
provide a way, for example, to create a software environment
consistent with Python 3.11.8 and another that is consistent with
Python 3.12.4 without conflict.  

Of course, like anything humans make, miniconda3 is not
perfect. There are times when the only solution is to delete an
environment and rebuild by reinstalling the desired packages.

### Miniconda3

After installing miniconda3, it is a good idea to update conda using the command
```bash
conda update conda
```
#### Step 1 
Assuming conda is properly installed and initialized on your laptop, you can create an environment, here called *mlphysics* using the command
```bash
conda create --name mlphysics
```
and activate it by doing
```bash
conda activate mlphysics
```
You need create the environment only once, but you must activate the desired environment whenever you create a new terminal window.

#### Step 2 
First install **pytorch**. (Tip: search the web for **conda install** and the
module name to get the exact syntax just in case it has changed.)
```
	conda install pytorch –c pytorch
```

#### Step 3
Install *jupyterlab*, *matplotlib*, *scikit-learn*, etc.
```bash
	conda install jupyterlab notebook
	conda install matplotlib
	conda install scikit-learn
	conda install pandas
	conda install sympy
	conda install imageio
	conda install tqdm
```
Again be sure to check the exact syntax. This does change from time to time!

#### Step 4
Install __git__ if it is not yet on your system, then download the **mlinphysics** package.
```bash
	conda install git
	mkdir tutorials
	cd tutorials
	git clone https://github.com/hbprosper/mlinphysics
```
In the above the GitHub package *mlinphysics* has been downloaded into a directory called *tutorials*.

#### Step 5

Open a new terminal window, navigate to the folder (directory) that contains **mlinphysics** and run  **jupyter lab** in that window (in blocking mode).
```bash
	jupyter lab
```
If all goes well, the jupyter lab environment will appear in your default browser. 
Navigate to the *mlinphysics* folder and under the *Files* menu item, click on the notebook *test.ipynb* and execute it. This notebook tries to import several Python modules. If it does so without error messages, you are ready to try out the other notebooks.

In general, it is better to work in another folder and to copy the pieces of *milinphysics* you wish to work with. That way you avoid conflicts if you need to update *mlinphysics* by navigating to that folder and doing 
```bash
	git pull
```
to download the latest version of this package.

