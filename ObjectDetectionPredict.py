"""
Script to get object detection predictions on raspberry pi
"""

__all__ = ['ObjectDetectionPredict']
__version__ = '0.1'
__author__ = 'NanoNets'

import os
import sys
import argparse

import tensorflow as tf
import numpy as np

from PIL import Image
from object_detection.utils import label_map_util, visualization_utils

parser = argparse.ArgumentParser(description='Predict result using trained graph')
parser.add_argument('--model', help='Model path')
parser.add_argument('--labels', help='Label file path')
parser.add_argument('--images', nargs='+', help='Path to images')
args = parser.parse_args()

current_dir = os.path.dirname(os.path.realpath(__file__))
root_dir, _ = os.path.split(current_dir)

sys.path.append(os.path.join(root_dir, 'models'))
MAX_CLASSES = 90

class ObjectDetectionPredict():
    def __init__(self, model_path, labels_path):
        label_map = label_map_util.load_labelmap(labels_path)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=MAX_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
        self.load_tf_graph(model_path)

    def load_tf_graph(self, model_path):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.detection_graph)
        return 0


    def detect_objects(self, image_np):
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent the level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        # Actual detection.
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        # Visualization of the results of a detection.
        visualization_utils.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=8)
        return scores, classes, image_np

def main():
    ObjectDetectionPredict_class = ObjectDetectionPredict(args.model, args.labels)
    for image_path in args.images:
        image = Image.open(image_path)
        (im_width, im_height) = image.size
        image_np = np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

        scores, classes, image_with_labels = ObjectDetectionPredict_class.detect_objects(image_np)
        print("\n".join("{0:<20s}: {1:.1f}%".format(ObjectDetectionPredict_class.category_index[c]['name'], s*100.) for (c, s) in zip(classes[0], scores[0])))

    ObjectDetectionPredict_class.sess.close()


if __name__ == '__main__':
   main()