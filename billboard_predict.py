from sample import input_init
import numpy as np
import sys
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
        rank = np.argmax(h)+1
        return rank

    def train(self, X, Y, num_iter,verbose):
        if verbose == False:
            print 'Training neural network...'
        for i in xrange(num_iter):
            a1 = X
            z2 = a1 * self.Theta1.T
            a2 = self.__sigmoid(z2)
            z3 = a2 * self.Theta2.T
            a3 = self.__sigmoid(z3)
            #print z3
            output = a3

            if verbose == True:
                print 'Output iteration '+ str(i)
                print output
            else:
                print 'Training iteration '+ str(i)

            #backpropagation
            delta_3 = Y - output
            delta_2 = np.multiply(delta_3 * self.Theta2, self.__sigmoid_gradient(z2))

            #print 'delta2'
            #print delta_2

            Theta2_grad = np.matrix(np.zeros(self.Theta2.shape))
            Theta2_grad = delta_3.T * a2
            Theta1_grad = delta_2.T * a1
            Theta2_grad /= float(X.shape[0])
            Theta1_grad /= float(X.shape[0])

            #update parameters
            NeuralNetwork.Theta2 += Theta2_grad
            NeuralNetwork.Theta1 += Theta1_grad

            #print 'Incremented Theta1'
            #print NeuralNetwork.Theta1
            ##########delta_2.sum(axis=0, dtype='float')

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __sigmoid_gradient(self, x):
        return np.multiply(self.__sigmoid(x), 1-self.__sigmoid(x))

if __name__ == '__main__':
    if len(sys.argv) > 2:
        print 'Too Many Arguments; use -help for more info'
        exit()

    #print sys.argv[1]
    options = sys.argv[1]

    verbose = False
    train = False
    predict = False

    if 'help' in options:
        print '\n-v       verbose     use for outputting prediction after each training iteration'
        print '-t       train       train neural network and write parameters to file'
        print '-p       predict     predict current chart\'s performance next week and this week\'s chart (to test accuracy)'
        print '-help    help        displays this\n'
        exit()
    if 'v' in options:
        verbose = True
    if 't' in options:
        train = True
    if 'p' in options:
        predict = True

    if verbose == False and train == False and predict == False and 'help' not in options:
        print '\nInvalid options; -help for more information\n'
        exit()
    #load training data
    print 'Loading Training Data...'
    yfile = open('data/yfile.txt')
    ydata = yfile.read().replace('\n',';')
    ydata = ydata[:-1]

    xfile = open('data/xfile.txt')
    xdata = xfile.read().replace('\n',';')
    xdata = xdata[:-1]

    neural_network = NeuralNetwork()

    X = np.matrix(xdata)
    Y = np.matrix(ydata)

    if train == True:
        neural_network.train(X, Y, 10000, verbose)
        Theta1_file = 'data/theta1file.txt'
        theta1_file = open(Theta1_file, 'w')
        theta1_file.truncate()
        Theta2_file = 'data/theta2file.txt'
        theta2_file = open(Theta2_file, 'w')
        theta2_file.truncate()
        np.savetxt(theta1_file, NeuralNetwork.Theta1, fmt="%.8f", comments='')
        np.savetxt(theta2_file, NeuralNetwork.Theta2, fmt="%.8f", comments='')
        theta1_file.close()
        theta2_file.close()

    if verbose == True:
        print 'After Training Theta1'
        print NeuralNetwork.Theta1
        print 'After Training Theta2'
        print NeuralNetwork.Theta2

    if predict == True:
        Theta1_file = 'data/theta1file.txt'
        theta1_file = open(Theta1_file)
        theta1_data = theta1_file.read().replace('\n',';')
        theta1_data = theta1_data[:-1]

        Theta2_file = 'data/theta2file.txt'
        theta2_file = open(Theta2_file)
        theta2_data = theta2_file.read().replace('\n',';')
        theta2_data = theta2_data[:-1]

        NeuralNetwork.Theta1 = np.matrix(theta1_data)
        NeuralNetwork.Theta2 = np.matrix(theta2_data)
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


    #print 'Predicting [3 20 2 1]'
    #print neural_network.predict(np.matrix('91 20 90 1'))
    #print neural_network.predict(np.matrix('3 20 2 1')).shape
