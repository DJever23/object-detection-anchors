import numpy as np


def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    :param box: tuple or array, shifted to the origin (i. e. width and height)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])#取width最小值
    y = np.minimum(clusters[:, 1], box[1])#取height最小值
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
    #先判断x是否为0，是返回True，否则返回False，然后用np.count_nonzero()返回非零个数，如果非零个数>0，说明box里的宽或高有零值，则触发异常
        raise ValueError("Box has no area")

    intersection = x * y#最小的宽高相乘得到交集面积
    box_area = box[0] * box[1]#当前框面积
    cluster_area = clusters[:, 0] * clusters[:, 1]#随机抽取的25个框的面积

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
            返回：每个框与所有聚类中心点的iou取最大值，将这些最大值相加再取均值
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    :param boxes: numpy array of shape (r, 4)
    :return: numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))#返回(rows,k)形状的空数组
    last_clusters = np.zeros((rows,))#返回(rows,)的全零数组

    np.random.seed()#随机生成种子数

    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]#在rows中随机抽取数字组成(k,)的一维数组，作为k个聚类中心，不能取重复数字
    print("clusters id {}".format(clusters))

    iter_num = 1
    while True:
        print("Iteration: %d" % iter_num)
        iter_num += 1

        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)
            #计算第row个box与随机抽取的25个box的iou，用此公式计算第row个box与随机抽取的25个box之间的距离
        print('{}'.format(distances.shape))#(144027, 25)

        nearest_clusters = np.argmin(distances, axis=1)#按行取最小值索引,每一个框属于第几个聚类中心
        print('nearest_clusters',nearest_clusters)
        print('{}'.format(type(nearest_clusters)))#(144027,)

        if (last_clusters == nearest_clusters).all():#所有的返回值都为True才会执行，即当每个框属于某个聚类中心的索引不再更新时跳出循环
            break

        for cluster in range(k):
            print('len(boxes[nearest_clusters == cluster]):{}'.format(len(boxes[nearest_clusters == cluster])))#返回True的数量
            #print('boxes[nearest_clusters == cluster]:{}'.format(boxes[nearest_clusters == cluster]))
            #print('(nearest_clusters == cluster):{}'.format(nearest_clusters == cluster))
            #[False False False ...  True  True False]
            if len(boxes[nearest_clusters == cluster]) == 0:#
                print("Cluster %d is zero size" % cluster)
                # to avoid empty cluster
                clusters[cluster] = boxes[np.random.choice(rows, 1, replace=False)]#此聚类中心size为0时重新为当前位置随机选择一个聚类中心
                continue

            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)#dist=np.median,在列的方向上求中位数
            #clusters[cluster] = np.median(boxes[nearest_clusters == cluster], axis=0)
            print('clusters[cluster]:{}'.format(clusters[cluster]))#[0.015625   0.02635432]
            #print('clusters[cluster]:{}'.format(clusters[cluster]))

        last_clusters = nearest_clusters
        #返回的是每一个聚类中心重新计算中位数，反复迭代计算后的新聚类中心点

    return clusters
