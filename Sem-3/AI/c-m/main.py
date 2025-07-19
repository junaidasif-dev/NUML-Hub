actual = [0, 1, 1, 1, 1, 0, 0, 1, 1, 1]
predicted = [1, 0, 1, 0, 0, 1, 0, 1, 1, 1]

#from sys import displayhook
from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(actual, predicted)
print(confusion_matrix)
