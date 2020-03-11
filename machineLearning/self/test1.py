from keras.datasets import boston_housing
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# 波士顿房价 影响因素13个 一行数据13个特征，得出最后的一个房价数据
print(train_data.shape,train_targets.shape)
print(test_data.shape,test_targets.shape)
print(train_data[0])
