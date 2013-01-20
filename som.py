import numpy
import pandas
from sklearn.decomposition import RandomizedPCA

class Cluster():
    def __init__(self, x, y):        
        self.map = []
        self.n_neurons = x*y
        self.sigma = x
        self.template = numpy.arange(x*y)
        self.alpha = 0.6
        self.alpha_final = 0.1
        self.shape = [x,y]
        self.epoch = 0
        
    def train(self, X, iter, batch_size=1):
        if len(self.map) == 0:
            x,y = self.shape
            self.map = numpy.zeros((self.n_neurons, len(X[0])))
            self.labels = numpy.arange(len(X)).astype("S10")
            eigen = RandomizedPCA(10).fit_transform(X.T).T
            self.map[0] = eigen[0]
            self.map[y-1] = eigen[1]
            self.map[(x-1)*y] = eigen[2]
            self.map[x*y - 1] = eigen[3]
            for i in range(4, 10):
                self.map[numpy.random.randint(1, self.n_neurons)] = eigen[i]
        self.total = iter
        #alpha_decay == (0.1/a)**1/n
        self.alpha_decay = (self.alpha_final/self.alpha)**(1.0/self.total)
        self.sigma_decay = (numpy.sqrt(self.shape[0])/(4*self.sigma))**(1.0/self.total)
        
        samples = numpy.arange(len(X))
        numpy.random.shuffle(samples)
    
        for i in xrange(iter):
            idx = samples[i:i + batch_size]
            self.iterate(X[idx])
    
    def transform(self, X, plot = False):
        plt.ion()
        v = numpy.sum((self.map - X)**2, axis=1).reshape(self.shape[0], self.shape[1])
        if plot:
            plt.contourf(v, alpha=.7)
        plt.draw()
        return v.ravel()
    
    def predict(self, X):
        #X = X.T
        x = self.shape[0]
        y = self.shape[1]
        dists = numpy.sum((self.map - X)**2, axis=1)
        idx = numpy.argmin(dists)
        return [idx%y, idx/x]
     
    def iterate(self, vector):  
        x, y = self.shape
        delta = self.map - vector
        dists = numpy.sum((delta)**2, axis=1).reshape(x,y)
        idx = numpy.argmin(dists)
        print "Epoch ", self.epoch, ": ", (idx/x, idx%y), "; Sigma: ", self.sigma, "; alpha: ", self.alpha
        
        self.sigma = self.sigma*self.sigma_decay
        dist_map = self.template.reshape(x,y)        
        dists = numpy.sqrt((dist_map/x - idx/x)**2 + (numpy.mod(dist_map,x) - idx%y)**2)
        
        h = numpy.exp(-(dists/self.sigma)**2)      
         
        for cell in xrange(len(self.map)):
            self.map[cell] = self.map[cell] - self.alpha*h[cell%x, cell/y]*delta[cell]
       
        
        #0.1 = a*d**n --> d**n = 0.1/a --> d == (0.1/a)**1/n
        self.alpha = self.alpha*self.alpha_decay
        
        self.epoch = self.epoch + 1 
        
    def save(self, filename="som_weights.csv"):
        data = numpy.concatenate([self.shape, [len(self.data[0])], self.map.ravel()])
        pandas.DataFrame(data).to_csv(filename)
        
    def load(self):
        data = pandas.read_csv("som_weights.csv").values[:,1].ravel()
        self.shape = data[[0,1]]
        z = data[2]
        self.map = data[3:].reshape(self.shape[0]*self.shape[1], z)
    
from PIL import Image
from tools import make_tile
import gzip
import cPickle

def load_mnist():   
    f = gzip.open("mnist/mnist.pkl.gz", 'rb')
    train, valid, test = cPickle.load(f)
    f.close()  
    return train[0]
    
def demo():
    # Get data
    X = load_mnist()
    cl = Cluster(20, 20)
    cl.train(X, 2000)  
    W = cl.map
    W = make_tile(W, img_shape= (28,28), tile_shape=(20,20))
    img = Image.fromarray(W)
    img.save("som_batch_size_5.png")
    
if __name__ == '__main__':
    demo()
