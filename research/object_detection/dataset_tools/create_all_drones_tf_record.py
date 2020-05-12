# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    python object_detection/dataset_tools/create_all_drones_tf_records.py \
        --output_path=/home/user/pascal.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os

from lxml import etree
import PIL.Image
import tensorflow as tf
from object_detection.utils import visualization_utils as vis_util
import pdb 
import numpy as np 
import matplotlib.pyplot as plt 

from tf1_tf2_compat_functions import tf_app, tf_io, tf_gfile
## tf.app
#if int(tf.__version__[0]) == 2:
#  tf_app = tf.compat.v1.app
#  tf_io  = tf.io
#else:
#  tf_app = tf.app
#  tf_io  = tf.python_io 


from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


flags = tf_app.flags
flags.DEFINE_string('raw_data_dir', '', 'Root directory to the images. \
    the full path of an image abc.jpg is supposed to be \
    raw_data_dir + "folder" given in the .xml file  \
    + "image_subdirectory" (which by default is Images) \
    + "filename" given in the .xml file' \
    )
flags.DEFINE_string('data_dir', '', 'Root directory to where we want to store the tfrecord files and other .txt files')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                    'merged set.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    'Root path to annotations directory.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'data/pascal_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                     'difficult instances')
flags.DEFINE_boolean('is_debug', False, 'Whether to visualize img + bbox')

FLAGS = flags.FLAGS

SETS = ['train', 'val']


def display_bbox(np_img, bboxes_array, classes_array, scores_array, category_index):
  '''
  * Args:
      np._img: uint8 numpy array with shape (img_height, img_width, 3)
      boxes: a numpy array of shape [N, 4]
      classes: a numpy array of shape [N]. Note that class indices are 1-based,
        and match the keys in the label map.
      scores: a numpy array of shape [N] or None.  If scores=None, then
        this function assumes that the boxes to be plotted are groundtruth
        boxes and plot all boxes as black with no classes or scores.
      category_index: a dict containing category dictionaries (each holding
        category index `id` and category name `name`) keyed by category indices.
  '''
  vis_util.visualize_boxes_and_labels_on_image_array( 
      np_img, \
      bboxes_array, \
      classes_array, \
      scores_array, \
      category_index,\
      use_normalized_coordinates=True,\
      line_thickness=8)
  
  plt.imshow(np_img)
  plt.savefig('visualize_img_bbox.png')

def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_path,
                       ignore_difficult_instances=False,
                       image_subdirectory='Images',
                       is_debug=False):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    dataset_directory: Path to root directory holding PASCAL dataset
    label_map_path: the prototxt file that contains a map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    image_subdirectory: String specifying subdirectory within the
      PASCAL dataset directory holding the actual image data.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  label_map_dict = label_map_util.get_label_map_dict(label_map_path)
  img_path = os.path.join(data['folder'], image_subdirectory, data['filename'])
  full_path = os.path.join(dataset_directory, img_path)
  with tf_gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')


  key = hashlib.sha256(encoded_jpg).hexdigest()

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  if 'object' in data:
    for obj in data['object']:
      difficult = bool(int(obj['difficult']))
      if ignore_difficult_instances and difficult:
        continue

      difficult_obj.append(int(difficult))

      xmin.append(float(obj['bndbox']['xmin']) / width)
      ymin.append(float(obj['bndbox']['ymin']) / height)
      xmax.append(float(obj['bndbox']['xmax']) / width)
      ymax.append(float(obj['bndbox']['ymax']) / height)
      classes_text.append(obj['name'].encode('utf8'))
      classes.append(label_map_dict[obj['name']])
      truncated.append(int(obj['truncated']))
      poses.append(obj['pose'].encode('utf8'))

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }))
  
  
  if is_debug:
    # Each box is     ymin, xmin, ymax, xmax = box in [0, 1]    
    bboxes_array   = np.array([ymin, xmin, ymax,  xmax])
    bboxes_array   = np.transpose(bboxes_array)
    classes_array  = np.array(classes)
    scores_array   = None 
    category_index = label_map_util.create_category_index_from_labelmap(\
                label_map_path, use_display_name=True)
    display_bbox(np.array(image), bboxes_array, classes_array, scores_array, category_index)
  return example


def main(_):
  if FLAGS.set not in SETS:
    raise ValueError('set must be in : {}'.format(SETS))
  

  writer = tf_io.TFRecordWriter(FLAGS.output_path)


  logging.info('Reading from all_drones %s dataset!')
  files_list_file = os.path.join(FLAGS.data_dir, FLAGS.set + '.txt') 

  examples_list = dataset_util.read_examples_list(files_list_file)
  for idx, example in enumerate(examples_list):
    if idx % 100 == 0:
      logging.info('On image %d of %d', idx, len(examples_list))
      logging.info('Save the tfrecord file to %s!'%FLAGS.output_path)
    path = os.path.join(FLAGS.annotations_dir, example + '.xml')
    with tf_gfile.GFile(path, 'r') as fid:
      xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

    tf_example = dict_to_tf_example(data, FLAGS.raw_data_dir, FLAGS.label_map_path,
                                    FLAGS.ignore_difficult_instances,\
                                    is_debug=FLAGS.is_debug)
    writer.write(tf_example.SerializeToString())

  writer.close()

if __name__ == '__main__':
  tf_app.run()

