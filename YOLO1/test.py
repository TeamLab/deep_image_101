import tensorflow as tf
import numpy as np
import cv2
import yolo.config as cfg
from yolo.yolo_net import YOLONet



class Detector(object):

    def __init__(self, net, weight_file):
        self.net = net
        self.weights_file = weight_file
        self.classes = cfg.CLASSES
        self.num_class = len(self.classes) # small_yolo is 20 class
        self.image_size = cfg.IMAGE_SIZE# image size is 448, 448, 3
        self.cell_size = cfg.CELL_SIZE  # grid cell is 7 by 7
        self.boxes_per_cell = cfg.BOXES_PER_CELL # B is 2 (only make 2 bounding box)
        self.threshold = cfg.THRESHOLD # threshold is 0.2 if each bounding box score is lower than 0.2 , set 0
        self.iou_threshold = cfg.IOU_THRESHOLD # IoU threshold is 0.5
        self.boundary1 = self.cell_size * self.cell_size * self.num_class # 7 X 7 X 20 = 980
        self.boundary2 = self.boundary1 + self.cell_size * self.cell_size * self.boxes_per_cell # 980 + 98 = 1078
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        print 'Restoring weights from: ' + self.weights_file
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.weights_file)

    # img = frame result = result
    def draw_result(self, img, result):
        for i in range(len(result)):
            x = int(result[i][1])
            y = int(result[i][2])
            w = int(result[i][3] / 2)
            h = int(result[i][4] / 2)

            # ex)  x = 323 y= 183 w =151 h = 115
            cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(img, (x - w, y - h - 20),
                          (x + w, y - h), (125, 125, 125), -1)
            cv2.putText(img, result[i][0] + ' : %.2f' % result[i][5], (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.CV_AA)

    # def camera_detector(self, cap, wait=10): result = self.detect(frame)
    def detect(self, img):
        img_h, img_w, _ = img.shape # test image shape is 360, 640, 3

        inputs = cv2.resize(img, (self.image_size, self.image_size))
        inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)
        inputs = (inputs / 255.0) * 2.0 - 1.0  # set the image's pixel value 0~1
        inputs = np.reshape(inputs, (1, self.image_size, self.image_size, 3))  # batch size , height , width , channel

        result = self.detect_from_cvmat(inputs)[0] # move detect_from_cvmat

        print 'length of result ', len(result)
        print 'shape of result ', np.shape(result)
        # if one object is detect , result shape is (1,6) class ,x, y, w, h, c

        for i in range(len(result)):
            result[i][1] *= (1.0 * img_w / self.image_size) # x
            result[i][2] *= (1.0 * img_h / self.image_size) # y
            result[i][3] *= (1.0 * img_w / self.image_size) # w
            result[i][4] *= (1.0 * img_h / self.image_size) # h

        return result

    def detect_from_cvmat(self, inputs):
        net_output = self.sess.run(self.net.logits,
                                   feed_dict={self.net.images: inputs})

        # net_output shape is (1,1470) S X S X ( B X 5 + C )

        results = []
        for i in range(net_output.shape[0]):
            results.append(self.interpret_output(net_output[i]))

        return results

    def interpret_output(self, output):
        probs = np.zeros((self.cell_size, self.cell_size,
                          self.boxes_per_cell, self.num_class)) # 7,7,2,20
        class_probs = np.reshape(output[0:self.boundary1], (self.cell_size, self.cell_size, self.num_class))
        scales = np.reshape(output[self.boundary1:self.boundary2], (self.cell_size, self.cell_size, self.boxes_per_cell))
        boxes = np.reshape(output[self.boundary2:], (self.cell_size, self.cell_size, self.boxes_per_cell, 4))
        offset = np.transpose(np.reshape(np.array([np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
                                         [self.boxes_per_cell, self.cell_size, self.cell_size]), (1, 2, 0))

        boxes[:, :, :, 0] += offset
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, 0:2] / self.cell_size
        boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])

        boxes *= self.image_size

        for i in range(self.boxes_per_cell):
            for j in range(self.num_class):
                probs[:, :, i, j] = np.multiply(
                    class_probs[:, :, j], scales[:, :, i])

        filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        boxes_filtered = boxes[filter_mat_boxes[0],
                               filter_mat_boxes[1], filter_mat_boxes[2]]
        probs_filtered = probs[filter_mat_probs]
        classes_num_filtered = np.argmax(filter_mat_probs, axis=3)[filter_mat_boxes[
            0], filter_mat_boxes[1], filter_mat_boxes[2]]

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]

        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0:
                continue
            for j in range(i + 1, len(boxes_filtered)):
                if self.iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):
            result.append([self.classes[classes_num_filtered[i]], boxes_filtered[i][0], boxes_filtered[
                          i][1], boxes_filtered[i][2], boxes_filtered[i][3], probs_filtered[i]])

        print 'current result ', result
        # example result = ['tvmonitor', 221.21773, 226.93465, 124.78653, 170.67261, 0.39018410444259644] class, x, y, w, h, c
        return result

    def iou(self, box1, box2):
        tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
            max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
            max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        if tb < 0 or lr < 0:
            intersection = 0
        else:
            intersection = tb * lr
        return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)

    def camera_detector(self, cap, wait=10):

        ret, _ = cap.read()

        while ret:
            ret, frame = cap.read()
            result = self.detect(frame)
            self.draw_result(frame, result)
            cv2.imshow('Camera', frame)
            cv2.waitKey(wait)
            ret, frame = cap.read()

def main():

    yolo = YOLONet(False)
    weight_file = 'YOLO_small.ckpt' # model directory
    detector = Detector(yolo, weight_file)
    # detect from camera
    cap = cv2.VideoCapture('test.mp4') # test video
    detector.camera_detector(cap)


if __name__ == '__main__':
    main()
