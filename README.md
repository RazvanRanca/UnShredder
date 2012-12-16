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
* Presentation 1 - project description and goals
* Poster Abstract
* Presentation 2 - cost function and heuristics analysis. Given probabilistic cost the bottleneck is the search heuristic. 

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
* Probabilistic cost functions done by normalizing probabilities over a source piece(assumes there must be a single correct source-destination match)
* Analysis of the cascading effect a mistake has with different search heuristics. Both heuristics suffer heavily because they cannot correct a mistake once made.
* Preliminary analysis of cost functions. Probabilistic cost seems to perform best. 

Up Next:
==
Reading:
--
* OCR cost functions
* ILP methods
* CPLEX

Writing:
--
* Ten-page interim report

Coding/Experiments:
--
* Refactor
* Analysis of the cascading effect on new search heuristics
* Analysis of including higher level features in cost function (eg: row positioning information)
* Developing an evaluation measure more closely correlated to how difficult it would be for a human to extract the relevant information from the reconstructed document
* Optimization of kruskal heuristic
* ILP approach
