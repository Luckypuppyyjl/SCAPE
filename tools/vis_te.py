import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from keras.models import Sequential,Model
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils,plot_model
from sklearn.model_selection import cross_val_score,train_test_split,KFold
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense,add,Input, Activation, Flatten, Convolution1D, Dropout,MaxPooling1D,BatchNormalization,GlobalAveragePooling1D,ZeroPadding1D,concatenate
from keras.models import load_model
from keras.models import model_from_json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from keras import layers
from keras.optimizers import SGD
# 载入数据
df = pd.read_csv(r'/data/yjliang/code/Category-Agnostic-Pose-Estimation/P2/Pose-for-Everything/work_dirs/token_3_bs16_spilt1_gong_256_test/test/test.csv')
X = np.expand_dims(df.values[:, 0:256].astype(float), axis=2)
Y = df.values[:, 256]
#Y[3999:]=Y[3999:]+20
Y[Y>20]=21
#Y=Y[:13536]
#13536
#X=X[:13536]
#这里很重要，不把标签做成独热码得形式最终出的图空白一片
encoder = LabelEncoder()
Y_encoded = encoder.fit_transform(Y)
Y_onehot = np_utils.to_categorical(Y_encoded)

X=X.reshape(17999,256)#这个形状根据读取到得csv文件得的大小自行调试，8621是列数，此列数比CSV文件中少一行
# 加载数据
def get_data():
	"""
	:return: 数据集、标签、样本数量、特征数量
	"""
	#digits = datasets.load_digits(n_class=10)
	digits=2
	data = X#digits.data		# 图片特征
	label = Y#digits.target		# 图片标签
	n_samples=17999#对应reshape中的行数
	n_features =256 #对应reshape中的列数
	return data, label, n_samples, n_features

cc=[(0.1, 0.2, 0.5),(0.1, 0.3, 0.5),(0.1, 0.4, 0.5),(0.1, 0.5, 0.5),
     (0.3, 0.2, 0.6),(0.3, 0.6, 0.7),(0.3, 0.1, 0.8),(0.9, 0.5, 0.9),
	(0.4, 0.2, 0.1), (0.4, 0.3, 0.2), (0.4, 0.4, 0.3), (0.4, 0.5, 0.4),
	(0.5, 0.5, 0.5), (0.6, 0.5, 0.5), (0.6, 0.4, 0.6), (0.6, 0.5, 0.7),
	(0.8, 0.2, 0.5), (0.8, 0.3, 0.5), (0.8, 0.4, 0.5), (0.8, 0.5, 0.5),(0.8, 0.7, 0.5),(0, 0, 0)
	]
# 对样本进行预处理并画图
def plot_embedding(data, label, title):
	"""
	:param data:数据集
	:param label:样本标签
	:param title:图像标题
	:return:图像
	"""
	x_min, x_max = np.min(data, 0), np.max(data, 0)
	data = (data - x_min) / (x_max - x_min)		# 对数据进行归一化处理
	fig = plt.figure()		# 创建图形实例
	ax = plt.subplot(111)		# 创建子图，经过验证111正合适，尽量不要修改
	# 遍历所有样本
	for i in range(data.shape[0]):
		# 在图中为每个数据点画出标签
		if int(label[i])<21:
		  plt.text(data[i, 0], data[i, 1], str(label[i]), color=cc[int(label[i])],
				 fontdict={'weight': 'bold', 'size': 7})
	plt.xticks()		# 指定坐标的刻度
	plt.yticks()
	plt.title(title, fontsize=14)
	# 返回值
	plt.show
	return fig


# 主函数，执行t-SNE降维
def main():
	data, label , n_samples, n_features = get_data()		# 调用函数，获取数据集信息
	print('Starting compute t-SNE Embedding...')
	ts = TSNE(n_components=2, init='pca', random_state=0)
	# t-SNE降维
	reslut = ts.fit_transform(data)
	# 调用函数，绘制图像
	fig = plot_embedding(reslut, label, 't-SNE Embedding of digits')
	# 显示图像
	plt.show()


# 主函数
if __name__ == '__main__':
	main()
