import numpy as np
import tensorflow as tf
from sys import float_info
from getUsers import retreivePairData
# Referenced from https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/logistic_regression.py

learning_rate = 0.001
training_epochs = 25
batch_size = 128
display_step = 1
threshold = 0.5

# an epsilon to prevent nan loss
epsilon = float_info.epsilon


def get_model(n):
    # tf Graph Input
    x = tf.placeholder(tf.float32, [None, n])  # Sequence1 concat Sequence2
    y = tf.placeholder(tf.float32, [None, 1])  # 0 or 1 label

    # Set model weights
    W = tf.Variable(tf.zeros([n, 1]))
    b = tf.Variable(tf.zeros(1))

    # Construct model
    pred = tf.nn.sigmoid(tf.matmul(x, W) + b)  # Logistic Regression
    yhat = tf.cast(pred > threshold, tf.float32)

    ## TODO we can play with what loss we use
    # Minimize error using cross entropy
    cost = tf.reduce_sum(-(y * tf.log(pred + epsilon) + (1 - y) * tf.log(1 - pred + epsilon)))
    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    params = {}
    params["x"] = x
    params["y"] = y
    params["W"] = W
    params["b"] = b
    params["pred"] = pred
    params["yhat"] = yhat
    params["cost"] = cost
    params["optimizer"] = optimizer
    return params

def train_and_eval(train_x, train_y, test_x, test_y, model_name ="logistic.ckpt"):

    m, n = train_x.shape

    params = get_model(n)
    x = params["x"]
    y = params["y"]
    pred = params["pred"]
    yhat = params["yhat"]
    cost = params["cost"]
    optimizer = params["optimizer"]

    # Initialize the variables (i.e. assign their default value)
    init_g = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init_g)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(m / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs = train_x[i * batch_size:(i + 1) * batch_size]
                batch_ys = train_y[i * batch_size:(i + 1) * batch_size]

                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                              y: batch_ys})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if (epoch + 1) % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

        print("Training Finished!")
        # Save the variables to disk.
        save_path = saver.save(sess, "./tmp/" + model_name)
        print("Model saved in file: %s" % save_path)

        # Evaluate Model
        # Variables for Evaluation
        accuracy, acc_update = tf.metrics.accuracy(y, yhat)
        false_pos, fp_update = tf.metrics.false_positives(y, yhat)
        false_neg, fn_update = tf.metrics.false_negatives(y, yhat)
        precision, prec_update = tf.metrics.precision(y, yhat)
        recall, rec_update = tf.metrics.recall(y, yhat)

        sess.run(tf.local_variables_initializer())
        metrics = [accuracy, acc_update,
                   precision, prec_update,
                   recall, rec_update,
                   false_pos, fp_update,
                   false_neg, fn_update]
        sess.run(metrics, feed_dict={x: test_x, y: test_y})

        predictions = sess.run(yhat, feed_dict={x: test_x})
        false_neg_count = sess.run(false_neg)
        specificity = (sum(label == 0 for label in predictions) - false_neg_count)/sum(label == 0 for label in test_y)

        print("Accuracy: ", sess.run(accuracy))
        print("Precision: ", sess.run(precision)) # True positive / (true positive + false positive)
        print("Recall: ", sess.run(recall)) # True positive / Positive
        print("Specificity: ", specificity) # True negative/ Negative
        print("False Positive Count: ", sess.run(false_pos))
        print("False Negative Count: ", false_neg_count)
        print("Number of Validation Examples: ", len(test_y))

    return save_path


def test_attack(attack_data, model_name="logistic.ckpt"):
    m, n = attack_data.shape

    params = get_model(n)
    x = params["x"]
    pred = params["pred"]
    yhat = params["yhat"]

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Start training
    with tf.Session() as sess:
        saver.restore(sess, "./tmp/" + model_name)

        predictions = sess.run(pred, feed_dict={x: attack_data})
        successful_attacks = sum(label == 1 for label in predictions)
        print("Number of Successful Attacks: ", successful_attacks)
        print("Number of Validation Examples: ", len(attack_data))
        print("Attack Success Rate: ", successful_attacks/len(attack_data))


def main():
    # #load trainX, trainY
    trainX, trainY, valX, valY, testX, testY = retreivePairData("RUS95:5")
    train_and_eval(trainX, trainY, valX, valY)

    # # Attack model
    # attacks = np.genfromtxt('data/attack_1mean_all.csv', delimiter=",", skip_header=False)
    # test_attack(attacks)

if __name__ == '__main__':
    main()