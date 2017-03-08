'''
Predicting the final match result based on the half time score
Features used: { half time score, # total shots, # shots on target }

Author: Omkar Acharya






'''

from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def main():
    # Reading CSV files
    df1 = pd.read_csv('data/E2.csv')
    df2 = pd.read_csv('data/E3.csv')

    # Appending the above datasets
    df = df1.append(df2)

    # Removing NaN row
    df = df.drop(df.index[[380]], axis = 0)

    # Removing the columns which are not needed
    df = df.drop(df.columns[[0, 1, 2, 3, 9, 10]], axis = 1)

    # Training labels
    labels = df['FTR'].values

    # Removing the remaining irrelevant features
    df = df.drop(df.columns[[range(9, df.shape[1])]], axis = 1)
    df = df.drop(df.columns[[0, 1, 2]], axis = 1)

    # Training data
    data = df.values

    # Test data
    limit = int(0.8 * data.shape[0])
    test_data = data[limit:]
    test_labels = labels[limit:]
    train_data = data[:limit]
    train_labels = labels[:limit]

    # Gaussian Naive Bayes classifier
    clf = GaussianNB()
    clf.fit(train_data, train_labels)
    predicted_labels = clf.predict(test_data)
    print('Gaussian NB Accuracy: ' + str(100 * accuracy_score(test_labels, predicted_labels)))

    # Multinomial Naive Bayes classifier
    clf = MultinomialNB()
    clf.fit(train_data, train_labels)
    predicted_labels = clf.predict(test_data)
    print('Multinomial NB Accuracy: ' + str(100 * accuracy_score(test_labels, predicted_labels)))

    # SVM classifier
    clf = SVC(kernel = 'linear')
    clf.fit(train_data, train_labels)
    predicted_labels = clf.predict(test_data)
    print('SVM Accuracy: ' + str(100 * accuracy_score(test_labels, predicted_labels)))


if __name__ == '__main__':
    main()