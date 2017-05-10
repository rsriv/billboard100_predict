from sample import input_init
import numpy as np
import billboard as bb
import sys
import math
import datetime
import time

#X features are [lastPos    weeks    rank    change]
#X and Y are written to files
#To access:
#   f = open('yfile.txt')
#   data = f.read().replace('\n',';')
#   data = data[:-1]
#   mat = np.matrix(data)

#only run this to download training data
#input_init.init_input(6000)

BRACKET = 4*[int]
BRACKET[0] = 10
BRACKET[1] = 35
BRACKET[2] = 60
BRACKET[3] = 100

def roundup(x):
    if x <= 10 and x >=0:
        return 0
    elif x > 10 and x <= 35:
        return 1
    elif x > 35 and x <60:
        return 2
    else:
        return 3


def is_num(string):
    try:
        int(string)
        return True
    except ValueError:
        return False


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
            output_bin = np.matrix('')
            for j in range(0,output.shape[1]):
                row = output[j]
                row = row.T
                index = np.argmax(row)
                temp = np.matrix(np.zeros((100,1)))
                temp[index] = 1
                if output_bin.shape == (1,0):
                    output_bin = temp.T
                else:
                    np.concatenate((output_bin, temp.T))

            #Where AA is a binary matrix of output


            if verbose == True:
                J = float(np.sum(np.multiply(-1*Y, np.log(output)) - np.multiply(1-Y, np.log(1-output)))/X.shape[0])
                print 'Cost of Iteration '+ str(i)+' -- '+str(J)
                #print output
            else:
                print '.',

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
    start_time = time.time()
    if len(sys.argv) > 2:
        print '\nToo Many Arguments; use -help for more info\n'
        exit()
    elif len(sys.argv) < 2:
        print '\nNot Enough Arguments; use -help for more info\n'
        exit()
    #print sys.argv[1]
    options = sys.argv[1]

    verbose = False
    train = False
    predict = False
    get_data = False

    if 'help' in options:
        print '\n-v       verbose     output raw prediction after each training iteration'
        print '-t       train       train neural network and write parameters to file'
        print '-p       predict     predict current chart\'s performance next week and this week\'s chart (to test accuracy)'
        print '-d       get data    download up-to-date Billboard Chart history data'
        print '-help    help        display this\n'
        exit()
    if 'v' in options:
        verbose = True
    if 't' in options:
        train = True
    if 'p' in options:
        predict = True
    if 'd' in options:
        get_data = True
        print 'Fetching past Billboard Chart data. This will take some time...'
        input_init.init_input(6000)
        print '...Data download complete.'

    if verbose == False and train == False and predict == False and get_data == False and 'help' not in options:
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

    #Train option
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

    #Verbose option
    if verbose == True:
        print 'After Training Theta1'
        print NeuralNetwork.Theta1
        print 'After Training Theta2'
        print NeuralNetwork.Theta2

    #Predict option
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

        #Collect last week's chart and this week's chart

        chart_current = bb.ChartData('hot-100')
        chart_last = bb.ChartData('hot-100',chart_current.previousDate)
        date_current = datetime.datetime.strptime(chart_current.date, "%Y-%m-%d")
        delta = datetime.timedelta(days=7)
        date_next = delta + date_current
        X_current = np.matrix('')

        Y_last = []
        #print '\nThis week\'s prediction for week of '+str(date_current)
        for i in range (0,100):
            song_current = chart_current[i]
            song_last = chart_last[i]
            change_last = 0
            if is_num(song_last.change):
                change_last = int(song_last.change)
            x_last = np.matrix([[song_last.lastPos, song_last.weeks, song_last.rank, change_last]])
            prediction = roundup(neural_network.predict(x_last))
            #print str(i+1) + '. ' + str(song_last) + ' -- This Week\'s Prediction: ' + str(prediction) + '-' + str(prediction-9)
            if i%10 == 9:
                print ''
            Y_last.append(prediction)
                #print x_last
        #print Y_last

        total_valid = 0
        count = 0
        ten_valid = 10
        ten_count = 0
        validity = 100 * [False]
        print '\n\nNext week\'s prediction for week of '+str(date_next)
        for i in range (0,100):
            correct = False
            song_current = chart_current[i]
            change_current = 0
            if is_num(song_current.change):
                change_current = int(song_current.change)
            x_current = np.matrix([[song_current.lastPos, song_current.weeks, song_current.rank, change_current]])
            prediction = roundup(neural_network.predict(x_current))
            index = Y_last[song_current.lastPos - 1]
            if song_current.lastPos > 0 and song_current.lastPos <= 100:
                total_valid += 1
                index = Y_last[song_current.lastPos - 1]
                if  index > 0 and song_current.rank <= BRACKET[index] and song_current.rank > BRACKET[index-1] or index == 0 and song_current.rank < BRACKET[index]:
                    count += 1
                    correct = True
                    validity[song_current.lastPos - 1] = True

                #top ten accuracy calculation
                if song_current.rank<=10:
                    if  index == 0:
                        ten_count += 1

            low = 0
            high = 0
            if prediction == 0:
                low = 1
                high = 10
            else:
                low = BRACKET[prediction - 1]
                high = BRACKET[prediction]

            if verbose == False or correct == False:
                print str(i+1) + '. ' + str(song_current) + ' -- Next Week\'s Prediction: ' + str(low) + '-' + str(high)
            else:
                if correct == True:
                    print str(i+1) + '. ' + str(song_current) + ' -- Next Week\'s Prediction: ' + str(low) + '-' + str(high) + ' (Correctly predicted last week)'

            if i%10 == 9:
                print ''

        print '\nThis week\'s prediction for week of '+str(date_current)
        for i in range (0,100):
            song_current = chart_current[i]
            song_last = chart_last[i]
            change_last = 0
            if is_num(song_last.change):
                change_last = int(song_last.change)
            x_last = np.matrix([[song_last.lastPos, song_last.weeks, song_last.rank, change_last]])
            prediction = roundup(neural_network.predict(x_last))

            low = 0
            high = 0
            if prediction == 0:
                low = 1
                high = 10
            else:
                low = BRACKET[prediction - 1]
                high = BRACKET[prediction]

            if validity[i] == False:
                print str(i+1) + '. ' + str(song_last) + ' -- This Week\'s Prediction: ' + str(low) + '-' + str(high)
            else:

                print str(i+1) + '. ' + str(song_last) + ' -- This Week\'s Prediction: ' + str(low) + '-' + str(high) + ' (Correct) '
            if i%10 == 9:
                print ''

        #accuracy statistics
        accuracy = float(count)/total_valid*100
        ten_accuracy = float(ten_count)/ten_valid*100
        #print 'count '+str(count)
        #print 'total_valid' + str(total_valid)
        print 'This Week\'s accuracy: ' + str(accuracy)+'%.\n'
        print 'This Week\'s Top Ten accuracy: ' + str(ten_accuracy)+'%.\n'
    print("--- done in %s seconds ---\n" % (time.time() - start_time))
        # print 'Predicted [3 20 2 1]'
        # print neural_network.predict(np.matrix('3 20 2 1'))
        #
        # print 'Predicted [99 30 98 1]'
        # print neural_network.predict(np.matrix('99 30 98 1'))
        #
        # print 'Predicted [70 1 21 49]'
        # print neural_network.predict(np.matrix('70 1 21 49'))
        #
        # print 'Predicted [66 12 76 -10]'
        # print neural_network.predict(np.matrix('66 12 76 -10'))
        #
        # print 'Predicted [20 8 70 -50]'
        # print neural_network.predict(np.matrix('20 8 70 -50'))
        #
        #print 'Predicted Something Just Like This by The Chainsmokers'
        #print neural_network.predict(np.matrix('5 9 8 -3'))


    #print 'Predicting [3 20 2 1]'
    #print neural_network.predict(np.matrix('91 20 90 1'))
    #print neural_network.predict(np.matrix('3 20 2 1')).shape
