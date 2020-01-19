From sickit Learn 0.22 -- Built on 01/2020

module includes classes and functions to call and visualize performance of 3 models
and the fusion call VotingRegressor or VotingClassifier. it gives the possibility 
of adjusting the vote with weights according to the performances of the various models

This is to simplify access to the functions created by sickit learn see :

https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html


This module call sickit learn and built with version == 0.22.1
you need pandas, numpy, matplotlib.

Votingregressor was add to sickit learn on 0.22

from sklearnMaster import voter

voter().get_model() return the fusion operate by Voting methode 

if one of model have better score then the Voting, call it for classical sickit learn to have acces
of all hyper parameters. I am working on function to add it on the module.

See demo from Notebook. I use joblib, but you can use other one if you wish.