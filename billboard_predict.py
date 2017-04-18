from sample import input_init
import numpy as np
#X features are [lastPos    weeks    rank    change]
#X and Y are written to files
#To access:
#   f = open('yfile.txt')
#   data = f.read().replace('\n',';')
#   data = data[:-1]
#   mat = np.matrix(data)

#only run this to download training data
#input_init.init_input(6000)


class NeuralNetwork:
    Theta1 = np.matrix(np.random.rand(25, 4))
    Theta2 = np.matrix(np.random.rand(1, 25))

    def __init__(self):
        np.random.seed(1)
        self.Theta1 = np.matrix(np.random.rand(25, 4))
        self.Theta2 = np.matrix(np.random.rand(1, 25))

    def predict(self, X):
        #forward propogation
        a1 = X
        z2 = a1 * self.Theta1.T
        a2 = self.__sigmoid(z2)
        z3 = a2 * self.Theta2.T
        h = z3
        return h

    def train(self, X, Y, num_iter):
        for i in xrange(num_iter):
            a1 = X
            z2 = a1 * self.Theta1.T
            a2 = self.__sigmoid(z2)
            z3 = a2 * self.Theta2.T
            print z3
            output = z3
            delta_3 = Y - output
            delta_2 = np.multiply(np.float64(delta_3*self.Theta2), np.float64(self.__sigmoid_gradient(z2)))
            Theta2_grad = delta_3.T * a2
            Theta1_grad = delta_2.T * a1
            Theta2_grad /= float(X.shape[0])
            Theta1_grad /= float(X.shape[0])

            self.Theta2 += Theta2_grad
            self.Theta1 += Theta1_grad

            print 'Incremented Theta1'
            print self.Theta1
            ##########delta_2.sum(axis=0, dtype='float')

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __sigmoid_gradient(self, x):
        return np.multiply(np.float64(x), np.float64((1 - x)))

if __name__ == '__main__':
    #load training data
    yfile = open('yfile.txt')
    ydata = yfile.read().replace('\n',';')
    ydata = ydata[:-1]

    xfile = open('xfile.txt')
    xdata = xfile.read().replace('\n',';')
    xdata = xdata[:-1]

    neural_network = NeuralNetwork()

    print 'Randomly Initialized Theta1'
    print NeuralNetwork.Theta1
    print 'Randomly Initialized Theta2'
    print NeuralNetwork.Theta2

    X = np.matrix(xdata)
    Y = np.matrix(ydata)

    neural_network.train(X, Y, 10)

    print 'After Training Theta1'
    print NeuralNetwork.Theta1
    print 'After Training Theta2'
    print NeuralNetwork.Theta2

    print 'Predicting [6 2 3 3]'
    print neural_network.predict(np.matrix('6 2 3 3'))

    print X
    print Y
