import numpy as np
import time
import random
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

# oversample method choice
SMOTE_CONSTANT = 1
ADASYN_CONSTANT = 2

# undersample method choice
RUS_CONSTANT = 3

def loadSingleData(filename):
    data = np.loadtxt(open(filename,'rb'), delimiter=',', skiprows=1)
    m, n = data.shape
    labels = data[:, 0]
    data_points = data[:, 1:]
    return labels, data_points

def resample_data(x, y, sample_choice = RUS_CONSTANT):
    if sample_choice == SMOTE_CONSTANT:
        sm = SMOTE(random_state=42)
        x, y = sm.fit_sample(x, y)
    elif sample_choice == ADASYN_CONSTANT:
        ada = ADASYN(random_state=42)
        x, y = ada.fit_sample(x, y)
    elif sample_choice == RUS_CONSTANT:
        rus = RandomUnderSampler(random_state=42)
        x, y = rus.fit_sample(x, y)
    return x, y

#loads data with oversampling
def loadPairData(filename, user_cutoff=5, example_cutoff=5, sample_choice = RUS_CONSTANT):
    data = np.loadtxt(open(filename,'rb'),delimiter=',', skiprows=1)
    m,n = data.shape

    pairData = []
    labels = []

    for a_index, a_data in enumerate(data):
        for b_index, b_data in enumerate(data):
            if a_data[0] < user_cutoff and b_data[0] < user_cutoff:
                if a_index % 51 < example_cutoff and b_index % 51 < example_cutoff:
                    if a_index != b_index:
                        # pairData.append((a_index, b_index))
                        concat = np.concatenate((a_data[1:], b_data[1:]))
                        pairData.append(concat)
                        if a_data[0] == b_data[0]: # If they are the same user
                            labels.append(1),
                        else:
                            labels.append(0)

    # resample data
    x_resampled, y_resampled = resample_data(pairData, labels, sample_choice=sample_choice)

    labeledData = list(zip(x_resampled, y_resampled))
    random.shuffle(labeledData)
    train_cutoff = int(len(labeledData)*0.90)
    val_cutoff = int(len(labeledData)*0.95)

    trainX, trainY = zip(*labeledData[:train_cutoff])
    trainX = np.asarray(trainX)
    trainY = np.asarray(trainY)

    valX, valY = zip(*labeledData[train_cutoff:val_cutoff])
    valX = np.asarray(valX)
    valY = np.asarray(valY)

    testX, testY= zip(*labeledData[val_cutoff:])
    testX = np.asarray(testX)
    testY = np.asarray(testY)

    return trainX, trainY, valX, valY, testX, testY

def retreivePairData(suffix):
    trainX = np.genfromtxt('data/trainX'+suffix+'.csv',delimiter=",",skip_header=False)
    trainY = np.genfromtxt('data/trainY'+suffix+'.csv',delimiter=",",skip_header=False)
    valX = np.genfromtxt('data/valX'+suffix+'.csv',delimiter=",",skip_header=False)
    valY = np.genfromtxt('data/valY'+suffix+'.csv',delimiter=",",skip_header=False)
    testX = np.genfromtxt('data/testX'+suffix+'.csv',delimiter=",",skip_header=False)
    testY = np.genfromtxt('data/testY'+suffix+'.csv',delimiter=",",skip_header=False)
    len_train = trainX.shape[0]
    len_val = valX.shape[0]
    len_test = testX.shape[0]

    return trainX, trainY.reshape((len_train, 1)), valX, valY.reshape((len_val, 1)), testX, testY.reshape((len_test, 1))


def singleUserData(userID=1):
    filename = 'keystroke.csv'
    labels, data_points = loadSingleData(filename=filename)
    single_user_label = labels == userID
    x, y = resample_data(data_points, single_user_label)
    labeledData = list(zip(x, y))
    random.shuffle(labeledData)
    x, y = zip(*labeledData)
    np.savetxt('data/singleUserX' + str(userID) + '.csv', x, delimiter=',')
    np.savetxt('data/singleUserY' + str(userID) + '.csv', y, delimiter=',')

def retreiveSingleUserData(userID=1):
    x = np.genfromtxt('data/singleUserX' + str(userID) + '.csv', delimiter=",", skip_header=False)
    y = np.genfromtxt('data/singleUserY' + str(userID) + '.csv', delimiter=",", skip_header=False)
    m = x.shape[0]

    return x, y.reshape((m, 1))

def main():
    start = time.time()
    filename = 'keystroke.csv'

    trainX, trainY, valX, valY, testX, testY = loadPairData(filename, user_cutoff=100, example_cutoff=100, sample_choice=RUS_CONSTANT)
    suffix = "RUS_90_5_5"
    np.savetxt('data/trainX' + suffix + '.csv', trainX, delimiter=',')
    np.savetxt('data/trainY' + suffix + '.csv', trainY, delimiter=',')
    np.savetxt('data/valX' + suffix + '.csv', valX, delimiter=',')
    np.savetxt('data/valY' + suffix + '.csv', valY, delimiter=',')
    np.savetxt('data/testX' + suffix + '.csv', testX, delimiter=',')
    np.savetxt('data/testY' + suffix + '.csv', testY, delimiter=',')

    # labels, data_points = loadSingleData(filename)

    end = time.time()
    print("Time to run:" + str(end-start))

if __name__ == '__main__':
    main()
    # singleUserData()

