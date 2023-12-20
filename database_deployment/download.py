import tensorflow as tf
import math, re, os
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import IPython.display as display
#from kaggle_datasets import KaggleDatasets
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
print("Tensorflow version " + tf.__version__)
AUTO = tf.data.experimental.AUTOTUNE

files = ["C:/Users/X/Downloads/trainpj.tfrecord", "C:/Users/X/Downloads/traincS.tfrecord", "C:/Users/X/Downloads/trainIc.tfrecord"]

raw_dataset = tf.data.TFRecordDataset(files)

for raw in raw_dataset.take(1):
    print(repr(raw))