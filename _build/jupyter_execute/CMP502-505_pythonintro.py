#!/usr/bin/env python
# coding: utf-8

# 
# # Introduction to Python
# 
# Short introduction to fundamental commands and structure of Python notebooks, as needed for this workbook.
# 

# ## General introduction
# 
# This workbook contains examples of simple Python commands and programs. These examples aim to enhance understanding of the maths components of CMP502 and CMP505. 
# Students can use Python to check their answers and visually see the mathematics in action. More broadly, a wide knowledge and ability to use other programming languages may help with problem solving and employability. 
# 
# Python is an open-source programming language used extensively around the world. It is also vital for carrying out Scientific Computation.
# 
# This workbook comprises of a series of Jupyter notebooks.
# Jupyter and python can be launched through the Anaconda data science platform on an individual machine. Anaconda can be downloaded here:
# 
# [https://www.anaconda.com](https://www.anaconda.com)
# 
# Jupyter notebooks combines sections of markdown text (which can include mathematical expressions) with executable elements of (Python) code. The executable elements appear in this workbook as grey boxes; when the workbook was created, *all* code elements were ran, and any output from them is seen directly below them.
# 
# The big advantage of this workbook is this:
# 
# **You can download each notebook to your computer or to the cloud and run all the code elements for yourself!**
# 
# I encourage you to read through each page of the workbook, then choose one of two options:
#  
# ![rocket](https://github.com/Abertay-University-SDI/MAT201_Python/blob/main/rocket.png?raw=1)
# ![download](https://github.com/Abertay-University-SDI/MAT201_Python/blob/main/down.png?raw=1)
# 
# 1.   click the rocket symbol  at the top of the page and launch the page in Google Colab.
# 2.   click the arrow symbol  at the top of the page, and download the notebook as a '.ipynb' file, in order to run the examples using jupyter-notebook.
# 
# I recommend using [Google Colab](https://colab.research.google.com/), unless you are confident that you have installed anaconda and jupyter notebook correctly on your machine. You will need a Google account to perform cloud computing, but executing the code elements is nearly identical in both Jupyter and Colab.

# ## Executing Jupyter notebook code
# 
# Notebooks in Jupyter look like this: ![Jupyter screenshot](https://github.com/Abertay-University-SDI/MAT201_Python/blob/main/JupyterNotebook_my_first_plot.png?raw=1)
# 
# Note that they comprise of markdown cells and code cells. This text is a markdown cell. Below, you can see my first code cell, containing the classic first command:

# In[1]:


print("Hello, World")


# The output should always appear below each code cell (if there is any).
# 
# Running the cell requires using SHIFT + ENTER. (In Colab, you may also click the play button next to each code cell to run it).
# 
# If you hit ENTER without the shift, the notebook will open the cell for editing, or perform a carriage return within that cell instead. You can edit code or text, then run it with SHIFT + ENTER.
# 
# Try modifying this statement yourself: **Python is ....**
# 
# I encourage you to edit copies of notebooks (to augment your lecture notes) and use SAVE AS (in Colab or Jupyter) to store your own personal copy of them to your own computer or OneDrive. 

# Python has extended functionality through libraries containing lots of extra commands, functions, and useful stuff. We import these libraries and commands using the *import* command:

# In[2]:


import time
time.sleep(3)


# In this case, we have imported a library called `time` in order to run a command called `time.sleep`. This command runs for an amount of time given by its argument (in our case, 3 seconds) and produces no output. Note that when the code is running the symbol within the square brackets changes (indicating it is running).
# 
# One thing to note is that some libraries have long names. When running a command from that library, we have so also write the name of the library out again:  
# 
# ```
# # time.sleep()
# ```
# In some cases you will see that common libraries are imported 'as' a different name (primarily so that our commands are shorter in future!)
# 

# There are **many** detailed introductions to Jupyter notebooks, and Google Colab, including textbooks, online workshops, videos, notebooks, theatrical productions (well maybe)... most are available on the web, like
# [this one](https://www.dataquest.io/blog/jupyter-notebook-tutorial/). Youtube videos [like this one](https://youtu.be/RLYoEyIHL6A) also allow you to see this process as you would on your machine.
# 
# 
# 
# ---
# 
# 

# ## Applications to CMP502 and CMP505 content
# 
# In our courses, some of our focus will be on the mathematics used to create specific tools and techniques. The rest of this introduction will illustrate a Python library called [Sympy](https://www.sympy.org/en/index.html). Like other Python packages, it is well documented, and widely used. We can use it to solve many symbolic computation problems (solving systems of equations, calculus, etc).
# 
# Let's go ahead and load in the libraries we need: in this case, we'll focus solely on `sympy` (which, for simplicity, when calling, we will refer to as `sym`):
# 

# In[3]:


import sympy as sym


# Let's very briefly look at some capabilities of `sympy`. `sympy` has three data types, Real Integer and Rational. Rational numbers are probably better known as fractions. If we wanted to work with the fraction of one half, for example, we could write this as a real number $0.5$, or as a fraction:

# In[4]:


onehalf = sym.Rational(1,2)
print(onehalf)


# We can perform basic maths with symbols in the usual way (e.g. addition, subtraction, multiplication, division):

# In[5]:


print("4*onehalf=", 4 * onehalf)


# We can use symbolic constants like $\pi$, $e$ (as in the exponential) or indeed infinity:

# In[6]:


mypi = sym.pi
print(mypi.evalf())


# This is just for starters: we can build systems of equations if we tell Python that some symbols represent unknown numbers:

# In[7]:


x, y = sym.Symbol('x'), sym.Symbol('y')
print(x + 2 * x + onehalf)


# See how Python has simplified the above expression, and added together the x terms? We can get it to do more complicated algebra:

# In[8]:


sym.expand( (x + onehalf * y)**2 )


# Sympy has expanded this binomial, just like we would using pen and paper. It knows all the rules we do! It also knows lots about trigonometric functions. The `simplify` command can simplify expressions, using all of the maths knowledge in the library. We know that $\cos^2{(x)}+\sin^2{(x)}=1$; so does sympy:

# In[9]:


print( sym.cos(x) * sym.cos(x) + sym.sin(x) * sym.sin(x))
print( sym.simplify( sym.cos(x) * sym.cos(x) + sym.sin(x) * sym.sin(x)) )


# It can even perform calculus operations like differentiation and integration:

# In[10]:


sym.diff(onehalf * x ** 2, x)


# where the command has differentiated the first expression with respect to x. We can perform partial derivatives just as easily: lets say we need 
# 
# $$
# \frac{\partial}{\partial x} \left(x^3+y+axy\right), 
# $$
# 
# where $x$ and $y$ are independent variables:

# In[11]:


sym.diff(x ** 3 + y + onehalf * x * y , x)


# easy peasy! What about integration? We know how to integrate simple functions like 
# 
# $$
# \int \frac{1}{{x}} ~{\rm{d}}x=\ln{|x|}+c,
# $$ 
# 
# but so does Sympy:

# In[12]:


sym.integrate(1 / x, x)


# (also noting that sympy refers to natural log ($ln$) as `log`). Sympy can also solve equations or sets of equations in one line:

# In[13]:


solution = sym.solve((x + 5 * y - 2, -3 * x + 6 * y - 15), (x, y))
solution[x], solution[y]


# It can also solve matrix equations and tons of other clever things. However, this is plenty of information to begin to explore simple differential calculus that we'll encounter in our CMP502 and CMP505 maths lectures.

# In[ ]:




