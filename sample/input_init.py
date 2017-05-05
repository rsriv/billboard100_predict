import billboard as bb
import numpy as np

#X features are [lastPos    weeks    rank    change]
#X and Y are written to files
#To access:
#   f = open('yfile.txt')
#   data = f.read().replace('\n',';')
#   data = data[:-1]
#   mat = np.matrix(data)


def is_num(string):
    try:
        int(string)
        return True
    except ValueError:
        return False


def init_input(n):
    X = np.matrix('')
    Y = np.matrix('')
    chart_current = bb.ChartData('hot-100')
    count = 0
    Xfile = 'xfile.txt'
    xfile = open(Xfile, 'w')
    xfile.truncate()
    Yfile = 'yfile.txt'
    yfile = open(Yfile, 'w')
    yfile.truncate()

    while X.shape[0] < n:
        if count > 0:
            chart_current = bb.ChartData('hot-100', chart_current.previousDate)
        chart_last = bb.ChartData('hot-100', chart_current.previousDate)
        i = 0
        song_current = chart_current[i]

        while song_current.lastPos == 0 and i < 100:
            i += 1

        for j in range(i, 100):
            song_current = chart_current[j]
            y = np.matrix(np.zeros((100,1)))
            y[song_current.rank-1] = np.matrix('1')
            y = y.T
            if song_current.lastPos == 0:
                continue

            if Y.shape == (1, 0):
                Y = y
            else:
                Y = np.concatenate((Y, y))

            song_last = chart_last[song_current.lastPos-1]
            change = 0

            if is_num(song_last.change):
                change = int(song_last.change)

            x = np.matrix([[song_last.lastPos, song_last.weeks, song_last.rank, change]])

            if X.shape == (1, 0):
                X = x
            else:
                X = np.concatenate((X, x))
        count += 1
        #print X
        print Y.shape
    #print X.shape
    #print X
    print Y.shape
    #print X.shape
    np.savetxt(yfile, Y, fmt="%d", comments='')
    np.savetxt(xfile, X, fmt="%d", comments='')
    xfile.close()
    yfile.close()
