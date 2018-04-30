from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from sklearn import cross_validation
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras import models
from keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import pandas as pd
# fix random seed for reproducibility
np.random.seed(7)
import datetime
import pickle

class Moduler(object):
    def __init__(self, file_name, mode):

        if mode =='train':
            self.train_model(file_name)

        elif mode == 'test':
            self.test_model()

    def train_model(self, file_name):

        print("started training at:")
        print(datetime.datetime.now())

        # dataset = numpy.loadtxt(file_name, delimiter=",")
        dataset = pd.read_csv(file_name)
        data = dataset.fillna(0.0)
        X_tag = data.iloc[:, 0:-1]
        y_tag = data.iloc[:, -1]

        number_of_features = dataset.shape[1]
        number_samples = dataset.shape[0]

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_tag)

        svc = svm.SVC(kernel='linear')
        C_s = np.logspace(-10, 0, 10)

        # scores = list()
        # scores_std = list()
        # print(C_s)
        # for C in C_s:
        # print("C is: {}".format(C))
        # print(datetime.datetime.now())
        svc.C = 0.01 #C
        this_scores = cross_val_score(svc, X_scaled, y_tag, n_jobs=-1, cv=10, scoring='roc_auc')
        print(this_scores)
        print(np.mean(this_scores))
        print(np.std(this_scores))
        # scores.append(np.mean(this_scores))
        # scores_std.append(np.std(this_scores))

        # run create_tensorflow_network
        # self.create_tensorflow_network(X_tag, y_tag, number_of_features, number_samples)

        # X_ = X_tag.as_matrix()
        # Y_ = y_tag.as_matrix()

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        # kfold = model_selection.KFold(n_splits=10, random_state=7)
        # modelCV = LogisticRegression()
        #
        # scoring = 'accuracy'
        # results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
        # try:
        #     print("10-fold cross validation average roc_auc: %.3f" % (np.mean(scores)))
        #     print("10-fold cross validation roc_auc: {}".format(scores))
        # except:
        #     pass

        try:
            # save the model to file
            filename = 'svm_model.sav'
            pickle.dump(svc, open(filename, 'wb'))
        except:
            print("Can't save the model")

        print("end training at:")
        print(datetime.datetime.now())

    def test_model(self, json_data):
        pass

    # Create function returning a compiled network
    def create_network(self, number_of_features):
        # Start neural network
        network = models.Sequential()

        # Add fully connected layer with a ReLU activation function
        network.add(layers.Dense(units=16, activation='relu', input_shape=(number_of_features,)))

        # Add fully connected layer with a ReLU activation function
        network.add(layers.Dense(units=16, activation='relu'))

        # Add fully connected layer with a sigmoid activation function
        network.add(layers.Dense(units=1, activation='sigmoid'))

        # Compile neural network
        network.compile(loss='binary_crossentropy',  # Cross-entropy
                        optimizer='rmsprop',  # Root Mean Square Propagation
                        metrics=['accuracy'])  # Accuracy performance metric

        # Return compiled network
        return network

    def create_tensorflow_network(self, X_tag, y_tag, number_of_features, number_samples):

        X = tf.convert_to_tensor(X_tag.as_matrix(), dtype=tf.float32)
        Y = tf.convert_to_tensor(y_tag.as_matrix(), dtype=tf.float32)

        # Parameters
        learning_rate = 0.001
        training_epochs = 15
        batch_size = 100
        display_step = 1
        num_examples = len(X_tag)

        # Network Parameters
        n_hidden_1 = number_of_features  # 1st layer number of features
        n_hidden_2 = number_of_features  # 2nd layer number of features
        n_input = number_of_features
        n_classes = len(set(y_tag))

        # tf Graph input
        x = tf.placeholder("float", [None, n_input])
        y = tf.placeholder("float", [None, n_classes])

        # Create model
        def multilayer_perceptron(x, weights, biases):
            # Hidden layer with RELU activation
            layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
            layer_1 = tf.nn.relu(layer_1)
            # Hidden layer with RELU activation
            layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
            layer_2 = tf.nn.relu(layer_2)
            # Output layer with linear activation
            out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
            return out_layer

        # Store layers weight & bias
        weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }

        # Construct model
        pred = multilayer_perceptron(x, weights, biases)

        # Define loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # Initializing the variables
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            # tf.initialize_all_variables().run()
            sess.run(init)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            for epoch in range(training_epochs):
                avg_cost = 0.
                total_batch = int(num_examples / batch_size)
                # Loop over all batches

                for i in range(total_batch):
                    batch_x = []
                    batch_y = []
                    for iteration in range(1, batch_size):
                        example, label = sess.run([X, Y])
                        batch_x.append(example)
                        batch_y.append(label)

                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                                  y: batch_y})
                    # Compute average loss
                    avg_cost += c / total_batch
                # Display logs per epoch step
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
            print("Optimization Finished!")
            coord.request_stop()
            coord.join(threads)

            saver = tf.train.Saver()
            saver.save(sess, 'finalized_model')

if __name__ == '__main__':
    Moduler("Master_Data_4.csv", mode='train')
