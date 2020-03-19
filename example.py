import glob
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib.pyplot as plt
from kmeans import kmeans, avg_iou

# ANNOTATIONS_PATH = "./data/pascalvoc07-annotations"
ANNOTATIONS_PATH = "./data/widerface-annotations"
CLUSTERS = 25
BBOX_NORMALIZE = False

def show_cluster(data, cluster, max_points=2000):
	'''
	Display bouding box's size distribution and anchor generated in scatter.散点图
	'''
	if len(data) > max_points:
		idx = np.random.choice(len(data), max_points)
		data = data[idx]#在所有的data中随机抽取max_points个数据
	plt.scatter(data[:,0], data[:,1], s=5, c='lavender')#输入数据是data的宽和高，s=5是点的大小，c是颜色
	plt.scatter(cluster[:,0], cluster[:, 1], c='red', s=100, marker="^")#‘^’是正三角形
	plt.xlabel("Width")
	plt.ylabel("Height")
	plt.title("Bounding and anchor distribution")
	plt.savefig("cluster.png")
	plt.show()

def show_width_height(data, cluster, bins=50):
	'''
	Display bouding box distribution with histgram.直方图
	'''
	if data.dtype != np.float32:
		data = data.astype(np.float32)
	width = data[:, 0]
	print('width_in show_width_height)',len(width))
	height = data[:, 1]
	print('height in show_width_height',height)
	ratio = height / width

	plt.figure(1,figsize=(20, 6))#num:图像编号或名称，数字为编号 ，字符串为名称；figsize:指定figure的宽和高，单位为英寸；
	plt.subplot(131)
	#subplot可以规划figure划分为n个子图，但每条subplot命令只会创建一个子图,131表示整个figure分成1行3列，共3个子图，这里子图在第一行第一列
	plt.hist(width, bins=bins, color='green')
	#width指定每个bin(箱子)分布的数据,对应x轴；bins这个参数指定bin(箱子)的个数,也就是总共有几条条状图；color指定条状图的颜色;默认y轴是个数
	plt.xlabel('width')
	plt.ylabel('number')
	plt.title('Distribution of Width')

	plt.subplot(132)
	plt.hist(height,bins=bins, color='blue')
	plt.xlabel('Height')
	plt.ylabel('Number')
	plt.title('Distribution of Height')

	plt.subplot(133)
	plt.hist(ratio, bins=bins,  color='magenta')
	plt.xlabel('Height / Width')
	plt.ylabel('number')
	plt.title('Distribution of aspect ratio(Height / Width)')
	plt.savefig("shape-distribution.png")
	plt.show()
	

def sort_cluster(cluster):
	'''
	Sort the cluster to with area small to big.
	'''
	if cluster.dtype != np.float32:
		cluster = cluster.astype(np.float32)
	print('cluster',cluster)
	area = cluster[:, 0] * cluster[:, 1]#计算每一个聚类中心点横纵坐标的乘积
	cluster = cluster[area.argsort()]#argsort函数返回的是数组值从小到大的索引值，此处将cluster按从小到大进行排序
	print('sorted cluster',cluster)
	ratio = cluster[:,1:2] / cluster[:, 0:1]
	print('ratio',ratio)
	return np.concatenate([cluster, ratio], axis=-1)  # 按轴axis连接array组成一个新的array,-1表示在最后一维进行合并，也就是行的方向合并


def load_dataset(path, normalized=True):
	'''
	load dataset from pasvoc formatl xml files
	'''
	dataset = []
	for xml_file in glob.glob("{}/*xml".format(path)):#获取path路径下所有的xml文件并返回一个list
		tree = ET.parse(xml_file)#调用parse()方法，返回解析树

		height = int(tree.findtext("./size/height"))
		width = int(tree.findtext("./size/width"))

		for obj in tree.iter("object"):
			if normalized:
				xmin = int(obj.findtext("bndbox/xmin")) / float(width)
				ymin = int(obj.findtext("bndbox/ymin")) / float(height)
				xmax = int(obj.findtext("bndbox/xmax")) / float(width)
				ymax = int(obj.findtext("bndbox/ymax")) / float(height)
			else:
				xmin = int(obj.findtext("bndbox/xmin")) 
				ymin = int(obj.findtext("bndbox/ymin")) 
				xmax = int(obj.findtext("bndbox/xmax")) 
				ymax = int(obj.findtext("bndbox/ymax"))
			if (xmax - xmin) == 0 or (ymax - ymin) == 0:
				continue # to avoid divded by zero error.
			dataset.append([xmax - xmin, ymax - ymin])

	return np.array(dataset)

print("Start to load data annotations on: %s" % ANNOTATIONS_PATH)
data = load_dataset(ANNOTATIONS_PATH, normalized=BBOX_NORMALIZE)
print('{}'.format(type(data)))#<class 'numpy.ndarray'>,(144027, 2)
print("Start to do kmeans, please wait for a moment.")
out = kmeans(data, k=CLUSTERS)#out为由kmeans找到的聚类中心点

out_sorted = sort_cluster(out)
print('out_sorted',out_sorted)
print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))#每个框与聚类中心点的最大IOU的平均值，可以用来表示所有框与聚类中心点的平均相似度

show_cluster(data, out, max_points=2000)

if out.dtype != np.float32:
	out = out.astype(np.float32)

print("Recommanded aspect ratios(width/height)")
print("Width    Height   Height/Width")
for i in range(len(out_sorted)):
	print("%.3f      %.3f     %.1f" % (out_sorted[i,0], out_sorted[i,1], out_sorted[i,2]))
show_width_height(data, out, bins=50)
