import numpy
print('ello1')
import pandas
print('ello2')
import sklearn
print('ello3')

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length','sepal-width','petal-length','petal-width' ,'class']
data_set = pandas.read_csv(url, names=names)

print(data_set.shape)
print('ello')


