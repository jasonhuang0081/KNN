from sklearn.preprocessing import MinMaxScaler

from arff import Arff
from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import train_test_split

from KNN import KNNClassifier

def normalize(train, test):
    scaler = MinMaxScaler()
    scaler.fit(train)
    scale_train = scaler.transform(train)
    scale_test = scaler.transform(test)
    return scale_train, scale_test

if __name__ == "__main__":
    ########### use for first 4 questions ###############
    # mat = Arff("diabetes.arff", label_count=1)
    # mat2 = Arff("diabetes_test.arff", label_count=1)
    # mat = Arff("seismic-bumps_train.arff", label_count=1)
    # mat2 = Arff("seismic-bumps_test.arff", label_count=1)
    # mat = Arff("magic telescope_train.arff", label_count=1)
    # mat2 = Arff("magic telescope_test.arff", label_count=1)
    mat = Arff("housing_train.arff", label_count=1)
    mat2 = Arff("housing_test.arff", label_count=1)
    raw_data = mat.data
    h, w = raw_data.shape
    train_data = raw_data[:, :-1]
    train_labels = raw_data[:, -1].reshape(-1, 1)

    raw_data2 = mat2.data
    h2, w2 = raw_data2.shape
    test_data = raw_data2[:, :-1]
    test_labels = raw_data2[:, -1].reshape(-1, 1)

    train_data, test_data = normalize(train_data,test_data)

    # KNN = KNNClassifier(labeltype='classification', weight_type='inverse_distance',k=15)
    # KNN = KNNClassifier(labeltype='classification', weight_type='', k=3)
    # KNN = KNNClassifier(labeltype='regression', weight_type='', k=15)
    KNN = KNNClassifier(labeltype='regression', weight_type='inverse_distance', k=3)
    KNN.fit(train_data, train_labels)
    pred, shape = KNN.predict(test_data)
    score = KNN.score(test_data, test_labels)
    print(score)
    # np.savetxt("diabetes_prediction.csv", pred, delimiter=',',fmt="%i")
    # np.savetxt("seismic-bump-prediction_mine.csv", pred, delimiter=',', fmt="%i")

    ################# the following is used for credit data set ##################
    # mat = Arff("credit.arff", label_count=1)
    # data = mat.data[:, 0:-1]
    # labels = mat.data[:, -1].reshape(-1, 1)
    # attr_type = mat.attr_types
    # KNN = KNNClassifier(labeltype='HEOM', weight_type='', k=3, columntype=attr_type)
    # X, X_test, y, y_test = train_test_split(data, labels, test_size=0.25)
    # KNN.fit(X, y)
    # score = KNN.score(X_test, y_test)
    # print(score)

