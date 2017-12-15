from getUsers import retreiveSingleUserData
from sklearn.cross_validation import StratifiedKFold
from keras.callbacks import LambdaCallback
import KerasModels as km
import trainKeras as tr


training_epochs = 30
threshold = 0.5
input_n = 142

def train_k_fold(model_fn, x, y, k=10):
    skf = StratifiedKFold(y.T[0], n_folds=k, shuffle=True)

    overall_metrics = [0, 0, 0, 0, 0]
    for i, (train, test) in enumerate(skf):
        print("Running Fold", i + 1, "/", k)
        model = model_fn()

        print("Training model")
        model.fit(x[train], y[train], epochs=training_epochs, batch_size=256, callbacks=[LambdaCallback(on_epoch_end=tr.clear_local_variables)])  # starts training

        metrics = model.evaluate(x[test], y[test])
        overall_metrics = [o + m/k for o, m in zip(overall_metrics, metrics)]

    # print(model_fn().metrics_names, overall_metrics)
    return overall_metrics

        # tr.train_and_eval(x[train], y[train], x[test], y[test], model, model_name=model_name+str(i+1)+"fold")


def main():
    x, y = retreiveSingleUserData()
    results = {}
    results["Logistic"] = train_k_fold(lambda: km.logistic_model(71), x, y)
    #
    # for i in range(1, 11):
    #     results["DeepNN"+str(i)] = train_k_fold(lambda: km.DeepNN(71, i), x, y)

    print(['loss', 'acc', 'recall', 'precision', 'false_negatives'])
    print(results)



if __name__ == '__main__':
    main()