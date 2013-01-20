Author: Mendrika Ramarlina
Email: m.ramarlina@gmail.com

Self-organizing map (SOM) is an unsupervised Neural Net technique that can be used to produce lower dimensional representation of the input space. Meaning, you can use SOM's to create a representation of input vectors
with less variables. In my example, I am using a SOM to reduce the dimensionality of handwritten digits image from 784 variables to 400 variables.

SOM uses competitive learning. For every example we present to the SOM, we improve only the group neurons that best represent that example. 
As we run through the whole dataset, the algorithm will make groups of neurons special in representing subsets of the data.

Initializing SOM:
    som = SOM(20, 20)

Training Model:
    som.train(X, 2000)  
        - X is the input vector
        - 2000 is the number of iterations
