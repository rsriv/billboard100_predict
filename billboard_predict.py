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
    def __init__(self):
        np.random.seed(1)
        NeuralNetwork.Theta1 = np.matrix(np.random.rand(20, 4))
        NeuralNetwork.Theta2 = np.matrix(np.random.rand(100, 20))

    def predict(self, X):
        #forward propogation
        a1 = X
        z2 = a1 * self.Theta1.T
        a2 = self.__sigmoid(z2)
        z3 = a2 * self.Theta2.T
        h = self.__sigmoid(z3)
        #CHANGE HERE TO get set max from output to 1 and rest to 0
        rank = np.argmax(h)+1
        return rank

    def train(self, X, Y, num_iter):
        for i in xrange(num_iter):
            a1 = X
            z2 = a1 * self.Theta1.T
            a2 = self.__sigmoid(z2)
            z3 = a2 * self.Theta2.T
            a3 = self.__sigmoid(z3)
            #print z3
            output = a3
            #CHANGE HERE TO get set max from output to 1 and rest to 0
            print 'Output iteration '+ str(i)
            print output

            delta_3 = Y - output
            delta_2 = np.multiply(delta_3 * self.Theta2, self.__sigmoid_gradient(z2))

            print 'delta2'
            print delta_2

            Theta2_grad = np.matrix(np.zeros(self.Theta2.shape))
            Theta2_grad = delta_3.T * a2
            Theta1_grad = delta_2.T * a1
            Theta2_grad /= float(X.shape[0])
            Theta1_grad /= float(X.shape[0])

            print 'Theta2_Grad'
            print Theta2_grad

            NeuralNetwork.Theta2 += Theta2_grad
            NeuralNetwork.Theta1 += Theta1_grad

            print 'Incremented Theta1'
            print NeuralNetwork.Theta1
            ##########delta_2.sum(axis=0, dtype='float')

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __sigmoid_gradient(self, x):
        return np.multiply(self.__sigmoid(x), 1-self.__sigmoid(x))

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
    t1 = NeuralNetwork.Theta1
    print 'Randomly Initialized Theta2'
    t2 = NeuralNetwork.Theta2

    X = np.matrix(xdata)
    Y = np.matrix(ydata)

    neural_network.train(X, Y, 10000)

    print 'After Training Theta1'
    print NeuralNetwork.Theta1
    print 'After Training Theta2'
    print NeuralNetwork.Theta2

    #print 'Predicting [3 20 2 1]'
    #print neural_network.predict(np.matrix('91 20 90 1'))
    #print neural_network.predict(np.matrix('3 20 2 1')).shape
    print 'Predicted [3 20 2 1]'
    print neural_network.predict(np.matrix('3 20 2 1'))

    print 'Predicted [99 30 98 1]'
    print neural_network.predict(np.matrix('99 30 98 1'))

    print 'Predicted [70 1 21 49]'
    print neural_network.predict(np.matrix('70 1 21 49'))

    print 'Predicted [66 12 76 -10]'
    print neural_network.predict(np.matrix('66 12 76 -10'))

    print 'Predicted [20 8 70 -50]'
    print neural_network.predict(np.matrix('20 8 70 -50'))

    print 'Predicted Something Just Like This by The Chainsmokers'
    print neural_network.predict(np.matrix('8 9 5 -3'))


    print 'theta_1'
    print t1

    print 'theta_2'
    print t2
