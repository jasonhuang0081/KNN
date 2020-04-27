from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler

from arff import Arff
from sklearn.model_selection import KFold
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


def normalize(train, test):
    scaler = MinMaxScaler()
    scaler.fit(train)
    scale_train = scaler.transform(train)
    scale_test = scaler.transform(test)
    return scale_train, scale_test

def wrapper(data, label, KNN):
    X, X_test, y, y_test = train_test_split(data, label, test_size=0.25)
    unselected_col = list(range(data.shape[1]))
    selected_col = []
    old_accuracy = 0
    while True:
        best_col = 0
        best_acc = 0
        for i in unselected_col:
            combinedIndex = selected_col + [i]
            # combinedData = np.concatenate((X[:,selected_col],X[:,i]),axis=1)
            combinedData = X[:,combinedIndex]
            combinedTesing = X_test[:,combinedIndex]
            KNN.fit(combinedData, y)
            score = KNN.score(combinedTesing, y_test)
            if score > best_acc:
                best_col = i
                best_acc = score
        selected_col.append(best_col)
        if best_acc - old_accuracy <= 0.02:
            break
        old_accuracy = best_acc
    combinedData = X[:, selected_col]
    combinedTesing = X_test[:, selected_col]
    KNN.fit(combinedData, y)
    score = KNN.score(combinedTesing, y_test)
    return selected_col, score

if __name__ == "__main__":

    # # mat = Arff("magic telescope_train.arff", label_count=1)
    # # mat2 = Arff("magic telescope_test.arff", label_count=1)
    # mat = Arff("housing_train.arff", label_count=1)
    # mat2 = Arff("housing_test.arff", label_count=1)
    # raw_data = mat.data
    # h, w = raw_data.shape
    # train_data = raw_data[:, :-1]
    # train_labels = raw_data[:, -1]
    #
    # raw_data2 = mat2.data
    # h2, w2 = raw_data2.shape
    # test_data = raw_data2[:, :-1]
    # test_labels = raw_data2[:, -1].reshape(-1, 1)
    #
    # train_data, test_data = normalize(train_data,test_data)
    #
    # # neigh = KNeighborsClassifier(n_neighbors=7,weights='distance',p=2,algorithm='auto')
    # neigh = KNeighborsRegressor(n_neighbors=5,weights='distance',p=2,algorithm='auto')
    #
    # neigh.fit(train_data, train_labels)
    # score = neigh.score(test_data,test_labels)
    #
    # print(score)

    ############### my own data set ####################
    mat = Arff("segment.arff", label_count=1)
    data = mat.data[:, 0:-1]
    labels = mat.data[:, -1]
    attr_type = mat.attr_types
    neigh = KNeighborsClassifier(n_neighbors=3)

    ## wrapper approach
    selected_col, train_score = wrapper(data,labels, neigh)       # call my wrapper function
    X, X_test, y, y_test = train_test_split(data[:,selected_col], labels, test_size=0.25)
    neigh.fit(X, y)
    score = neigh.score(X_test, y_test)
    print(score)

    ## PCA approach
    # pca = PCA(n_components=7)
    # pca.fit(data)
    # # print(pca.explained_variance_ratio_.cumsum())
    # new_data = pca.transform(data)
    #
    # X, X_test, y, y_test = train_test_split(new_data, labels, test_size=0.25)
    # neigh.fit(X, y)
    # score = neigh.score(X_test, y_test)
    # print(score)