import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class KNNClassifier(BaseEstimator,ClassifierMixin):


    def __init__(self,labeltype='classification',weight_type='inverse_distance', columntype=[],k=3): ## add parameters here
        """
        Args:
            columntype for each column tells you if continues[real] or if nominal.
            weight_type: inverse_distance voting or if non distance weighting. Options = ["no_weight","inverse_distance"]
        """
        self.attr_type = columntype
        self.weight_type = weight_type
        self.k = k
        self.type = labeltype

    def getDistance(self,instance):

        diff = instance - self.data
        dist = np.linalg.norm(diff,axis=1)
        return dist.reshape(1,-1)

    def fit(self,data,labels):
        """ Fit the data; run the algorithm (for this lab really just saves the data :D)
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        (values, counts) = np.unique(labels[:, 0], return_counts=True)
        self.labelTypes = len(values)
        self.data = data
        self.labels= labels
        self.max_value_list = np.nanmax(data,axis=0)
        self.min_value_list = np.nanmin(data,axis=0)
        return self

    def doRegression(self, data):
        result = np.ones((data.shape[0], 1))
        for i in range(data.shape[0]):
            distance = self.getDistance(data[i, :])
            if np.count_nonzero(distance==0) > 0:
                vector, index = np.where(distance == 0)
                result[i,0] = self.labels[index[0]]
            else:
                if self.weight_type == 'inverse_distance':
                    invDistance = np.square(np.reciprocal(distance))
                    index_k_largest = np.argpartition(invDistance, -self.k)
                    labels_k = self.labels[index_k_largest[0,-self.k:]]
                    dist_k = invDistance[0,index_k_largest[0,-self.k:]]
                    sum = 0
                    for j in range(labels_k.shape[0]):
                        sum = sum + labels_k[j,0]*dist_k[j]
                    result[i,0] = sum/np.sum(dist_k)
                else:
                    index_k_smallest = np.argpartition(distance, self.k)
                    labels_k = self.labels[index_k_smallest[0,:self.k]]
                    result[i,0] = np.sum(labels_k)/len(labels_k)
        return result, result.shape

    def d(self,x,y,attr_type,col):
        if np.isnan(x) or np.isnan(y):
            return 1
        elif attr_type == "continuous":
            return np.abs(x-y)/(self.max_value_list[col] - self.min_value_list[col])
        elif attr_type == "nominal":
            if x == y:
                return 0
            else:
                return 1
        else:
            print("exception")

    def HEOM(self,instance):
        distance = np.ones((1, self.data.shape[0]))
        for i in range(self.data.shape[0]):
            sum = 0
            for j in range(self.data.shape[1]):
                sum = sum + (self.d(self.data[i][j],instance[j],self.attr_type[j],j))**2
            distance[0,i] = np.sqrt(sum)
        return distance

    def predict(self,data):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """


        if self.type == 'regression':
            result, shape = self.doRegression(data)
            return result, shape

        result = np.ones((data.shape[0],1))
        if self.type == 'HEOM':
            for i in range (data.shape[0]):
                distance = self.HEOM(data[i,:])
                index_k_smallest = np.argpartition(distance, self.k)
                labels_k = self.labels[index_k_smallest[0, :self.k]]
                (values, counts) = np.unique(labels_k, return_counts=True)
                ind = np.argmax(counts)
                result[i, 0] = values[ind]
            return result, result.shape

        for i in range(data.shape[0]):
            distance = self.getDistance(data[i,:])
            if np.count_nonzero(distance==0) > 0:
                vector, index = np.where(distance == 0)
                result[i,0] = self.labels[index[0]]
            else:
                if self.weight_type == 'inverse_distance':
                    invDistance = np.square(np.reciprocal(distance))
                    index_k_largest = np.argpartition(invDistance, -self.k)
                    labels_k = self.labels[index_k_largest[0,-self.k:]]
                    dist_k = invDistance[0,index_k_largest[0,-self.k:]]
                    comparison = np.zeros(self.labelTypes)
                    for j in range(labels_k.shape[0]):
                        comparison[int(labels_k[j,0])] = comparison[int(labels_k[j,0])]  + dist_k[j]
                    ind = np.where(comparison == np.amax(comparison))
                    result[i, 0] = ind[0]
                else:
                    index_k_smallest = np.argpartition(distance, self.k)
                    labels_k = self.labels[index_k_smallest[0,:self.k]]

                    (values, counts) = np.unique(labels_k, return_counts=True)
                    ind = np.argmax(counts)
                    result[i, 0] = values[ind]

        return result, result.shape

    #Returns the Mean score given input data and labels
    def score(self, x, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.
        Args:
                X (array-like): A 2D numpy array with data, excluding targets
                y (array-like): A 2D numpy array with targets
        Returns:
                score : float
                        Mean accuracy of self.predict(X) wrt. y.
        """
        output, shape = self.predict(x)
        row, col = output.shape
        if self.type == 'regression':
            MSR = np.sum(np.square(output - y)) / row
            return MSR
        else:
            correct = 0
            for i in range(row):
                if output[i,0] == y[i,0]:
                    correct += 1
            return correct/row


