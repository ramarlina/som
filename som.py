import numpy
from sklearn.decomposition import RandomizedPCA


class SOM():
    def __init__(self, x, y):        
        self.map = []
        self.n_neurons = x*y
        self.sigma = x
        self.template = numpy.arange(x*y).reshape(self.n_neurons,1)
        self.alpha = 0.6
        self.alpha_final = 0.1
        self.shape = [x,y]
        self.epoch = 0
        
    def train(self, X, iter, batch_size=1):
        if len(self.map) == 0:
            x,y = self.shape
            # first we initialize the map
            self.map = numpy.zeros((self.n_neurons, len(X[0])))
            
            # then we the pricipal components of the input data
            eigen = RandomizedPCA(10).fit_transform(X.T).T
            
            # then we set different point on the map equal to principal components to force diversification
            self.map[0] = eigen[0]
            self.map[y-1] = eigen[1]
            self.map[(x-1)*y] = eigen[2]
            self.map[x*y - 1] = eigen[3]
            for i in range(4, 10):
                self.map[numpy.random.randint(1, self.n_neurons)] = eigen[i]
                
        self.total = iter
        
        # coefficient of decay for learning rate alpha
        self.alpha_decay = (self.alpha_final/self.alpha)**(1.0/self.total)
        
        # coefficient of decay for gaussian smoothing
        self.sigma_decay = (numpy.sqrt(self.shape[0])/(4*self.sigma))**(1.0/self.total)
        
        samples = numpy.arange(len(X))
        numpy.random.shuffle(samples)
    
        for i in xrange(iter):
            idx = samples[i:i + batch_size]
            self.iterate(X[idx])
    
    def transform(self, X):
        # We simply compute the dot product of the input with the transpose of the map to get the new input vectors
        res = numpy.dot(numpy.exp(X),numpy.exp(self.map.T))/numpy.sum(numpy.exp(self.map), axis=1)
        res = res / (numpy.exp(numpy.max(res)) + 1e-8)
        return res
     
    def iterate(self, vector):  
        x, y = self.shape
        
        delta = self.map - vector
        
        # Euclidian distance of each neurons with the example
        dists = numpy.sum((delta)**2, axis=1).reshape(x,y)
        
        # Best maching unit
        idx = numpy.argmin(dists)
        print "Epoch ", self.epoch, ": ", (idx/x, idx%y), "; Sigma: ", self.sigma, "; alpha: ", self.alpha
        
        # Linearly reducing the width of Gaussian Kernel
        self.sigma = self.sigma*self.sigma_decay
        dist_map = self.template.reshape(x,y)     
        
        # Distance of each neurons in the map from the best matching neuron
        dists = numpy.sqrt((dist_map/x - idx/x)**2 + (numpy.mod(dist_map,x) - idx%y)**2).reshape(self.n_neurons, 1)
        #dists = self.template - idx
        
        # Applying Gaussian smoothing to distances of neurons from best matching neuron
        h = numpy.exp(-(dists/self.sigma)**2)      
         
        # Updating neurons in the map
        self.map -= self.alpha*h*delta
       
        # Decreasing alpha
        self.alpha = self.alpha*self.alpha_decay
        
        self.epoch = self.epoch + 1 
        
        
##################################################### 
#       EXAMPLE: TRAINING SOM ON MNIST DATA         #
#####################################################       
    
from PIL import Image
from tools import make_tile
import gzip
import cPickle

def load_mnist():   
    f = gzip.open("mnist/mnist.pkl.gz", 'rb')
    train, valid, test = cPickle.load(f)
    f.close()  
    return train[0][:20000],train[1][:20000]
    
def demo():
    # Get data
    X, y = load_mnist()
    cl = SOM(20, 20)
    cl.train(X, 2000)  
    
    # Plotting hidden units
    W = cl.map
    W = make_tile(W, img_shape= (28,28), tile_shape=(20,20))
    img = Image.fromarray(W)
    img.save("som_results.png")
    
    
    # creating new inputs
    X = cl.transform(X)
   
    # we can plot "landscape" 3d to view the map in 3d
    landscape = cl.transform(numpy.ones((1,28**2)))
    return cl.map, X, y,landscape
    
    
if __name__ == '__main__':
    demo()
