import pandas as pd
import os.path as osp
import sklearn.svm as svm
import pickle
from sklearn.multiclass import OneVsRestClassifier


data_root = "./data"


def get_train_data(split=15000):
	"""
	Returns:
		train_x, train_y, val_x, val_y
	"""
	data_file = osp.join(data_root, "TrainSamples.csv")
	label_file = osp.join(data_root, "TrainLabels.csv")
	data = pd.read_csv(data_file, header=None).to_numpy()
	label = pd.read_csv(label_file, header=None).to_numpy()
	return data[0:split], label[0:split].reshape(-1), data[split::], label[split::].reshape(-1)


def get_test_data():
	data_file = osp.join(data_root, "TestSamples.csv")
	data = pd.read_csv(data_file, header=None).to_numpy()
	return data


def train():
	train_x, train_y, val_x, val_y = get_train_data(split=19000)
	model = svm.SVC(kernel="poly", degree=3)
	print("Training the model ...")
	model = model.fit(train_x, train_y)
	print("Model Training is completed !")

	print("Validating the model ...")
	score = model.score(val_x, val_y)
	print("Acc: {:.6f}".format(score))
	# print(model.predict(val_x))

	return model


def test(model):
	test_data = get_test_data()
	results = model.predict(test_data)
	# results = results.reshape((1, -1))
	results.astype(int)
	results = pd.DataFrame(results)
	results.to_csv(osp.join(data_root, "TestLabels.csv"), header=None, index=False)


if __name__ == "__main__":
	# training stage
	model = train()
	pickle.dump(model, open("model.p", "wb"))
	# testing stage
	# model = pickle.load(open("model.p", "rb"))
	# test(model)