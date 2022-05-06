# sSVN
A package accompanying https://arxiv.org/abs/2204.09039, which implements Stein variational gradient descent (SVGD), Stein variational Newton (SVN), and their stochastic counterparts (sSVGD, sSVN).

# Motivation
Stein variational gradient descent (SVGD) is a general-purpose optimization-based sampling algorithm that has recently exploded in popularity, but is limited by two issues: it is known to produce
biased samples, and it can be slow to converge on complicated distributions. A recently proposed
stochastic variant of SVGD (sSVGD) addresses the first issue, producing unbiased samples by incorporating a special noise into the SVGD dynamics such that asymptotic convergence is guaranteed.
Meanwhile, Stein variational Newton (SVN), a Newton-like extension of SVGD, dramatically accelerates the convergence of SVGD by incorporating Hessian information into the dynamics, but
also produces biased samples. In this paper we derive, and provide a practical implementation of,
a stochastic variant of SVN (sSVN) which is both asymptotically correct and converges rapidly. We demonstrate
that this method holds promise for parameter estimation problems of modest dimension.

# Flow illustration
Code for these animations found in `notebooks/double_banana_flows.ipynb`
![SVGD](https://media.giphy.com/media/m9FLn6E31NDUfGNrEJ/giphy.gif)
![sSVGD](https://media.giphy.com/media/CH7cd0w3CbeaGon7vt/giphy.gif)
![SVN](https://media.giphy.com/media/lThEzFxUIC3z1q208Q/giphy.gif)
![sSVN](https://media.giphy.com/media/BCxYUmkAQtbEJd1vam/giphy.gif)


# Getting started
## Installation
Code has been tested using `Python 3.6` on both Linux and Windows. Install libraries in `requirements.txt` and run a notebook of your choice from `notebooks/`.
