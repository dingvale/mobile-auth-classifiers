from getUsers import retreivePairData
from keras.callbacks import LambdaCallback
import matplotlib.pyplot as plt
import numpy as np
from KerasModels import *

training_epochs = 20
threshold = 0.5
input_n = 142


def main():
    # #load trainX, trainY
    trainX, trainY, valX, valY, testX, testY = retreivePairData("RUS_90_5_5")

    models =[("Logistic", logistic_model(input_n)),
             ("DNN_1", DeepNN(input_n, 1)),
             ("DNN_2", DeepNN(input_n, 2)),
             ("DNN_3", DeepNN(input_n, 3)),
             ("DNN_4", DeepNN(input_n, 4)),
             ("DNN_5", DeepNN(input_n, 5)),
             ("DNN_6", DeepNN(input_n, 6)),
             ("DNN_7", DeepNN(input_n, 7)),
             ("DNN_8", DeepNN(input_n, 8)),
             ("DNN_9", DeepNN(input_n, 9)),
             ("DNN_10", DeepNN(input_n, 10)),
             ("Triangle", TrigangleDNN(input_n, 3))]

    metrics = []
    for name, model in models:
        model.load_weights("Models/"+name)
        print("-------------------------\n")
        print("Evaluating Model", name)
        metric = model.evaluate(x=valX, y=valY)
        metrics.append([name]+metric)

    # np.savetxt("modelResults.csv", np.asarray(metrics), delimiter=',')
    print(logistic_model(input_n).metrics)
    for metric in metrics:
        print(metric)

if __name__ == '__main__':
    main()