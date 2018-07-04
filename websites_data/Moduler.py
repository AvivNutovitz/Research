from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import pandas as pd
np.random.seed(7)
import datetime
import pickle
# Importing the Keras libraries and packages
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import cross_val_score, KFold


class Moduler(object):
    def __init__(self, y, scoring='accuracy', with_visual_features=True, num_folds=5):

        self.model = None
        self.results_folder = os.getcwd()+os.sep+'resulsts'
        self.y_for_the_model = y
        self.scoring_metric = scoring
        self.with_visual_features = with_visual_features
        self.num_folds = num_folds
        self.model = self.find_or_train_model()

    def find_or_train_model(self):
        """
        This function try to find the model that fit the users request.
        If it does not exist then it trained the model.
        :return: the model object
        """
        self.set_data()
        model = self.find_model()
        if model == None:
            return self.train_model()
        else:
            return model

    def find_model(self):
        """
        This function tries to find the model in the results folder,
        That will match y value and scoring method
        :return: the model if founded, else None
        """

        y_folder = "y_{}".format(self.y_for_the_model)
        main_path = r'' + self.results_folder + os.sep + y_folder + os.sep + self.scoring_metric + os.sep
        try:
            loaded_model = None
            if self.with_visual_features:
                for fold in range(1, self.num_folds + 1, 1):
                    # set the model file path
                    model_to_save_file_name = "dnn_model_with_visual_features_y_{}_fold_{}.json".format(self.y_for_the_model, fold)
                    # set the weights file path
                    model_weights_to_save_file_name = "dnn_model_weights_with_visual_features_y_{}_fold_{}.h5".format(self.y_for_the_model, fold)

                    # load json and create model
                    json_file = open(model_to_save_file_name, 'r')
                    loaded_model_json = json_file.read()
                    json_file.close()
                    loaded_model = model_from_json(loaded_model_json)

                    # load weights into new model
                    loaded_model.load_weights(model_weights_to_save_file_name)
            else:
                model_to_save_file_name = "logistic_regression_model_without_visual_features_y_{}.sav".format(self.y_for_the_model)
                loaded_model = pickle.load(open(model_to_save_file_name, 'rb'))

            print("Loaded model from disk")
            return loaded_model

        except Exception as e:
            print(e)
            return None

    def train_model(self):

        print("started training at:")
        print(datetime.datetime.now())

        model = None
        if self.with_visual_features:
            model = self.with_features()
        else:
            model = self.without_features()

        print("end training at:")
        print(datetime.datetime.now())
        return model

    def set_data(self):
        if self.with_visual_features:
            file_name = "Master_data_y_{}.csv".format(self.y_for_the_model)

            dataset = pd.read_csv(file_name, index_col='index')
            data = dataset.fillna(0.0)

            tag = 'y_{}'.format(self.y_for_the_model)
            X_tag = data.loc[:, data.columns != tag]

            self.y_tag = data[[tag]].as_matrix()
            scaler = MinMaxScaler()
            self.X_scaled = scaler.fit_transform(X_tag)

        else:
            file_name = "regular features.csv"

            data = pd.read_csv(file_name, index_col='index')
            feature_list = ['number_of_a',
                            'number_of_all_background_color_elements',
                            'number_of_all_color_elements',
                            'number_of_area_elements',
                            'number_of_different_tag_names',
                            'number_of_div',
                            'number_of_elements_from_file',
                            'number_of_elements_with_absolute_position',
                            'number_of_elements_with_relative_position',
                            'number_of_h1',
                            'number_of_h2',
                            'number_of_h3',
                            'number_of_img',
                            'number_of_span',
                            'number_of_text_elements',
                            'total_element_area_elements',
                            'total_number_of_words_elements',
                            'total_text_length_elements',
                            'total_text_total_rate_elements']

            tag = 'y_{}'.format(self.y_for_the_model)
            X_tag = data[feature_list]
            self.y_tag = data[[tag]].as_matrix()

            scaler = MinMaxScaler()
            self.X_scaled = scaler.fit_transform(X_tag)

            self.X_scaled = X_tag

    def test_model(self):

        if self.with_visual_features:
            for fold in range(1, self.num_folds+1, 1):
                X_test = pd.read_csv("X_test_data_with_visual_features_for_y_{}_fold_{}.csv".format(self.y_for_the_model, fold))
                y_test = pd.read_csv("y_test_data_with_visual_features_for_y_{}_fold_{}.csv".format(self.y_for_the_model, fold))

                self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[self.scoring_metric])

                # # Predicting the Test set results
                y_pred = self.model.predict(X_test)
                y_pred = (y_pred > 0.5)

                # Creating the Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)

                print("confusion_matrix:")
                print(cm)
                tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=y_pred).ravel()
                print("tn: {}, fp: {}, fn: {}, tp:{}".format(tn, fp, fn, tp))

                score = self.model.evaluate(X_test, y_test, verbose=0)
                print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))

        else:
            x_csv_file_name = "X_test_data_with_visual_features_for_y_{}.csv".format(self.y_for_the_model)
            y_csv_file_name = "y_test_data_with_visual_features_for_y_{}.csv".format(self.y_for_the_model)

    def with_features(self):
        '''Run your neural network and output your prediction probabilities'''

        # Initialize the roc-auc score running average list
        # Initialize a count to print the number of folds
        count = 0
        model = None
        # Initialize your cross vailidation
        # Set shuffle equals True to randomize your splits on your training data
        kf = KFold(n_splits=self.num_folds, random_state=41, shuffle=True)

        # Set up for loop to run for the number of cross vals you defined in your parameter
        for train_index, test_index in kf.split(self.X_scaled):
            count += 1
            print('Fold #: ', count)

            # This indexs your train and test data for your cross validation and sorts them
            # in random order, since we used shuffle equals True
            X_nn_train, X_nn_test = self.X_scaled[train_index], self.y_tag[test_index]
            y_nn_train, y_nn_test = self.X_scaled[train_index], self.y_tag[test_index]

            # Define your input dimension, which must equal the number of variables in your
            # training data. If it does not you will get a goofy error.
            input_dim = X_nn_train.shape[1]

            # Initialize your neural network structure we defined above to build your model
            print("Building model...")
            model = self.build_nn(input_dim)

            # Fit your model
            # You can select the number of epochs and and batch size you would like to use
            # for your neural network.
            print("Training model...")
            model.fit(X_nn_train, y_nn_train, batch_size=100, nb_epoch=30)

            try:
                model_json = model.to_json()
                model_to_save_file_name = "dnn_model_with_visual_features_y_{}_fold_{}.json".format(
                    self.y_for_the_model, count)
                with open(model_to_save_file_name, "w") as json_file:
                    json_file.write(model_json)
                # serialize weights to HDF5
                weights_to_save_file_name = "dnn_model_weights_with_visual_features_y_{}_fold_{}.h5".format(
                    self.y_for_the_model, count)
                model.save_weights(weights_to_save_file_name)
                print("Saved model to disk")
            except:
                pass

            try:
                # export X_test, y_test to csv files
                if self.with_visual_features:
                    x_csv_file_name = "X_test_data_with_visual_features_for_y_{}_fold_{}.csv".format(
                        self.y_for_the_model, count)
                    y_csv_file_name = "y_test_data_with_visual_features_for_y_{}_fold_{}.csv".format(
                        self.y_for_the_model, count)
                else:
                    x_csv_file_name = "X_test_data_without_visual_features_for_y_{}_fold_{}.csv".format(
                        self.y_for_the_model, count)
                    y_csv_file_name = "y_test_data_without_visual_features_for_y_{}_fold_{}.csv".format(
                        self.y_for_the_model, count)

                pd.DataFrame(X_nn_test).to_csv(x_csv_file_name, header=None, index=False)
                pd.DataFrame(y_nn_test).to_csv(y_csv_file_name, header=None, index=False)
            except:
                pass


    def without_features(self):

        model = svm.SVC(kernel='linear')
        model.C = 0.01  # C
        scores = cross_val_score(model, self.X_scaled, self.y_tag, n_jobs=-1, cv=5, scoring='roc_auc')
        print("roc_auc: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        try:
            # save the model to file
            filename = 'svm_model_without_visual_features_y_{}.sav'.format(self.y_for_the_model)
            pickle.dump(model, open(filename, 'wb'))
        except:
            print("Can't save the model")

    def build_nn(self, input_dim):
        '''Build your Neural Network structure'''

        # Initialize your sequential NN
        # You can change the model to something other than sequential, for more info check Keras' documentation.
        model = Sequential()

        # This part adds the number of perceptrons you would like to use in the first layer
        # The input dimension is the number of variables you have in your data
        # The activation parameter is what kind of function you to use for your perceptron function
        # You can use a variety of different perceptron functions but relu is very common
        model.add(Dense(output_dim=1000, input_dim=input_dim, init='uniform', activation='relu'))

        # Adding hidden layer
        model.add(Dense(output_dim=1000, init='uniform', activation='relu'))

        # Adding hidden layer
        model.add(Dense(output_dim=500, init='uniform', activation='relu'))

        # Adding hidden layer
        model.add(Dense(output_dim=500, init='uniform', activation='relu'))

        # Adding hidden layer
        model.add(Dense(output_dim=100, init='uniform', activation='relu'))

        # Adding hidden layer
        model.add(Dense(output_dim=50, init='uniform', activation='relu'))

        # The last layer is your output layer so the number of perceptions must be equal to the
        # amount target classes your data set has.
        # documentation.
        model.add(Dense(output_dim=1, init='uniform', activation='relu'))

        # Lastly you want to define your loss function, your optimizer and your metric for scoring.
        # This will vary based on your goals, but for a binary target this parameter configuration
        # works well.
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Return your neural network
        return model


if __name__ == '__main__':
    m0 = Moduler(y=1, with_visual_features=True)
    m1 = Moduler(y=1, with_visual_features=True)
    m6 = Moduler(y=6, with_visual_features=True)
    m8 = Moduler(y=8, with_visual_features=True)
    m14 = Moduler(y=14, with_visual_features=True)
    m19 = Moduler(y=19, with_visual_features=True)
    m21 = Moduler(y=21, with_visual_features=True)
