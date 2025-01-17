import numpy as np
import json
import glob
import xml.etree.ElementTree as ET
class YOLO_Kmeans:

  def __init__(self, cluster_number, filename):
    self.cluster_number = cluster_number
    self.filename = filename

  def iou(self, boxes, clusters):  # 1 box -> k clusters
    n = boxes.shape[0]
    k = self.cluster_number

    box_area = boxes[:, 0] * boxes[:, 1]
    box_area = box_area.repeat(k)
    box_area = np.reshape(box_area, (n, k))

    cluster_area = clusters[:, 0] * clusters[:, 1]
    cluster_area = np.tile(cluster_area, [1, n])
    cluster_area = np.reshape(cluster_area, (n, k))

    box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
    cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
    min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

    box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
    cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
    min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
    inter_area = np.multiply(min_w_matrix, min_h_matrix)

    result = inter_area / (box_area + cluster_area - inter_area)
    return result

  def avg_iou(self, boxes, clusters):
    accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
    return accuracy

  def kmeans(self, boxes, k, dist=np.median):
    box_number = boxes.shape[0]
    distances = np.empty((box_number, k))
    last_nearest = np.zeros((box_number,))
    np.random.seed()
    clusters = boxes[np.random.choice(box_number, k, replace=False)]  # init k clusters
    while True:
      distances = 1 - self.iou(boxes, clusters)

      current_nearest = np.argmin(distances, axis=1)
      if (last_nearest == current_nearest).all():
        break  # clusters won't change
      for cluster in range(k):
        clusters[cluster] = dist(boxes[current_nearest == cluster], axis=0)

      last_nearest = current_nearest

    return clusters

  def result2txt(self, data):
    f = open("yolo_anchors.txt", 'w')
    row = np.shape(data)[0]
    for i in range(row):
      if i == 0:
        x_y = "%d,%d" % (data[i][0], data[i][1])
      else:
        x_y = ", %d,%d" % (data[i][0], data[i][1])
      f.write(x_y)
    f.close()

  def txt2boxes(self):
    f = open(self.filename, 'r')
    dataSet = []
    for line in f:
      infos = line.split(" ")
      length = len(infos)
      for i in range(1, length):
        width = int(infos[i].split(",")[2]) - \
                int(infos[i].split(",")[0])
        height = int(infos[i].split(",")[3]) - \
                 int(infos[i].split(",")[1])
        dataSet.append([width, height])
    result = np.array(dataSet)
    f.close()
    return result
  def xml2boxes(self):
    self.filename=["/datasets/VOCdevkit/VOC2012/Annotations",
                   "/datasets/VOCdevkit/VOC2007/Annotations"]
    dataset = []
    for dir in self.filename:
      for xml_file in glob.glob("{}/*xml".format(dir)):
        tree = ET.parse(xml_file)

        height = int(tree.findtext("./size/height"))
        width = int(tree.findtext("./size/width"))
        for obj in tree.iter("object"):
          xmin = float(obj.findtext("bndbox/xmin")) / width*512
          ymin = float(obj.findtext("bndbox/ymin")) / height*512
          xmax = float(obj.findtext("bndbox/xmax")) / width*512
          ymax = float(obj.findtext("bndbox/ymax")) / height*512
          dataset.append([xmax - xmin, ymax - ymin])
    return np.array(dataset)

  def json2boxes(self):
    img2wh=json.load(open(self.filename,'r'))
    boxes=[]
    for k,v in img2wh.items():
      wlist=v[0]
      hlist=v[1]
      for w,h in zip(wlist,hlist):
        boxes.append([w,h])
    result=np.array(boxes)
    return result
  def json2clusters(self):
    all_boxes = self.json2boxes()
    import time
    s=time.time()
    result = self.kmeans(all_boxes, k=self.cluster_number)
    result = result[np.lexsort(result.T[0, None])]
    self.result2txt(result)
    print("K anchors:\n {}".format(result))
    print("Accuracy: {:.2f}%".format(
      self.avg_iou(all_boxes, result) * 100))
    print(time.time()-s)
  def txt2clusters(self):
    all_boxes = self.txt2boxes()
    result = self.kmeans(all_boxes, k=self.cluster_number)
    result = result[np.lexsort(result.T[0, None])]
    self.result2txt(result)
    print("K anchors:\n {}".format(result))
    print("Accuracy: {:.2f}%".format(
      self.avg_iou(all_boxes, result) * 100))
  def xml2clusters(self):
    all_boxes = self.xml2boxes()
    result = self.kmeans(all_boxes, k=self.cluster_number)
    result = result[np.lexsort(result.T[0, None])]
    self.result2txt(result)
    print("K anchors:\n {}".format(result))
    print("Accuracy: {:.2f}%".format(
      self.avg_iou(all_boxes, result) * 100))

if __name__ == "__main__":
  cluster_number = 9
  # filename = "2012_train.txt"
  # filename = "/home/gwl/PycharmProjects/mine/tf2-yolo3/dataset/coco_info.json"
  filename="/home/gwl/datasets/VOCdevkit/VOC2012/Annotations"
  kmeans = YOLO_Kmeans(cluster_number, filename)
  kmeans.xml2clusters()
  # kmeans.json2clusters()
  # kmeans.txt2clusters()
