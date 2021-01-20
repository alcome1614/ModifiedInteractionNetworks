# Simulation of physical systems with variants of the Interaction Network

The aim of this thesis is to primarily learn to predict trajectories of physical systems
by using Machine Learning. We have used the Interaction Network as a base
model and introduced tweaks to its structure in the framework of Graph Networks
in order to deal with systems without interaction, with force-based interactions and
systems governed by the Vicsek model. We have been able to replicate very well systems
without interaction and systems governed by a Vicsek model of infinite reach.
However the results with force-based systems are mediocre because they need more
trainable parameters and training data. The results for the Vicsek model with a finite
radius of reach are the worst but we have learned the necessity and the methodology
of introducing attention to the base model to deal with this class of problem.
