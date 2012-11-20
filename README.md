UnShredder - Work in progress
==========

Done:
==
Reading:
--
* Initial Lit. Review identifying state of research in area
* Local search methods
* Exploration of cost functions

Writing:
--
* Summary of papers regarding local search and cost functions
* Presentation 1
* Poster Abstract

Coding/Experiments:
--
* Bit to Bit and Gaussian Cost functions
* Above cost functions restricted to black bits
* Greedy1D search baseline
* Prim search baseline
* Gaussian simulation of edges cost
* Basic image processing tied into above baselines.
* REFACTOR
* Kruskal heuristic search function. Should be better than Graph2D or Prim.
* Solved several bottlenecks, Prim now works on 15x15 documents in ~15 seconds but kruskal still takes about a minute on 12x12.
* Vizualizations tools, can observe what algorithm does at every step
* Basic neural network cost function implemented, further work needed to get good results
* Cost function based on distribution of 6 pixel window taken over the shreds

Up Next:
==
Reading:
--
* OCR cost functions
* ILP methods
* CPLEX

Writing:
--
* Results of experiments done so far, effect of cost function

Coding/Experiments:
--
* Further testing of new cost functions
* Bayesian cost function
* Neural Net cost function
* Optimization of kruskal heuristic
* ILP approach
