<div>Author: Mendrika Ramarlina <br/>
Email: m.ramarlina@gmail.com</div>

<p>
Self-organizing map (SOM) is an unsupervised Neural Net technique that can be used to produce lower dimensional representation of the input space. Meaning, you can use SOM's to create a representation of input vectors
with less variables. In my example, I am using a SOM to reduce the dimensionality of handwritten digits image from 784 variables to 400 variables.
</p>
<p>
SOM uses competitive learning. For every example we present to the SOM, we improve only the group neurons that best represent that example. 
As we run through the whole dataset, the algorithm will make groups of neurons specialize themselves in representing subsets of the data.
</p>

Libraries I am using:
    <br/>- NumPy: scientific computing library for Python

Initializing SOM:<br/>
    som = SOM(20, 20)<br/><br/>

Training Model:<br/>
    som.train(X, 2000)  <br/>
        - X is the input vector<br/>
        - 2000 is the number of iterations<br/>
