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

yfile = open('yfile.txt')
ydata = yfile.read().replace('\n',';')
ydata = ydata[:-1]
Y = np.matrix(ydata)
xfile = open('xfile.txt')
xdata = xfile.read().replace('\n',';')
xdata = xdata[:-1]
X = np.matrix(xdata)

print X
print Y
