#coding:utf-8

##############
# 1 load module
##############

from __future__ import print_function
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='5'

import numpy as np
import tensorflow as tf
import random
import cv2
import time

from trainer import Trainer
from config import get_config
from data_loader import get_loader
from utils import prepare_dirs_and_logger, save_config

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import scipy.misc
import numpy as np
from glob import glob
from tqdm import trange
from itertools import chain
from collections import deque

from models import *
from utils import save_image
import utils_seg
import helpers

def next(loader):
    return loader.next()[0].data.numpy()

def to_nhwc(image, data_format):
    if data_format == 'NCHW':
        new_image = nchw_to_nhwc(image)
    else:
        new_image = image
    return new_image

def to_nchw_numpy(image):
    if image.shape[3] in [1, 3]:
        new_image = image.transpose([0, 3, 1, 2])
    else:
        new_image = image
    return new_image

def norm_img(image, data_format=None):
    #image = image/255.0
    image = image/127.5 - 1.
    if data_format:
        image = to_nhwc(image, data_format)
    return image

def denorm_img(norm, data_format):
    return tf.clip_by_value(to_nhwc(norm*255, data_format), 0, 255)
    # return tf.clip_by_value(to_nhwc((norm + 1)*127.5, data_format), 0, 255)

def reverse_one_hot(x):
    # x = tf.transpose(x,[0,2,3,1])
    # locations = tf.where(tf.equal(x, 1))
    # indices = locations[:,3]
    indices = tf.argmax(x,1)
    indices = tf.reshape(indices,(1,1,256,256))
    indices = tf.cast(indices,'float32')
    return indices

#def denorm_img2(x, data_format):
#    return np.clip(np.squeeze(x.transpose([0,2,3,1]))*255.0,0,255)

def slerp(val, low, high):
    """Code from https://github.com/soumith/dcgan.torch/issues/14"""
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high

def generate(sess, inputs, root_path=None, path=None, idx=None, save=True):
    x = sess.run(G, {z: inputs})
    if path is None and save:
        path = os.path.join(root_path, '{}_G.png'.format(idx))
        #save_image(x, path)
        cv2.imwrite(path, 255*np.clip(x[0,0:,:,:].transpose([1,2,0]),0,1))
        print("[*] Samples saved: {}".format(path))
        return x

def autoencode(sess, input_image, inputs, path, idx=None, x_fake=None):
    items = {
        'real': inputs,
        'fake': x_fake,
    }
    for key, img in items.items():
        if img is None:
            continue
        x_path = os.path.join(path, '{}_D_{}.png'.format(idx, key))
        x = sess.run(AE_x, {z: input_image, xx: img})
        #save_image(x, x_path)
        ae = np.clip(x[0].transpose([1,2,0]),0,255)
        cv2.imwrite(x_path, ae)
        print("[*] Samples saved: {}".format(x_path))

def encode(sess, inputs):
    if inputs.shape[3] in [1, 3]:
        inputs = inputs.transpose([0, 3, 1, 2])
    return sess.run(D_z, {x: inputs})

def decode(sess, z):
    return sess.run(AE_x, {D_z: z})

# Get a list of the training, validation, and testing file paths
def prepare_data(config):
    dataset_dir = config.dataset
    train_input_names=[]
    train_output_names=[]
    train_output_weight_names=[]
    val_input_names=[]
    val_output_names=[]
    test_input_names=[]
    test_output_names=[]
    for file in os.listdir(dataset_dir + "/train"):
        cwd = os.getcwd()
        train_input_names.append(cwd + "/" + dataset_dir + "/train/" + file)
    for file in os.listdir(dataset_dir + "/train_labels"):
        cwd = os.getcwd()
        train_output_names.append(cwd + "/" + dataset_dir + "/train_labels/" + file)
    if config.is_edge_weight:
        for file in os.listdir(dataset_dir + "/train_labels_weights"):
            cwd = os.getcwd()
            train_output_weight_names.append(cwd + "/" + dataset_dir + "/train_labels_weights/" + file)
    for file in os.listdir(dataset_dir + "/val"):
        cwd = os.getcwd()
        val_input_names.append(cwd + "/" + dataset_dir + "/val/" + file)
    for file in os.listdir(dataset_dir + "/val_labels"):
        cwd = os.getcwd()
        val_output_names.append(cwd + "/" + dataset_dir + "/val_labels/" + file)
    for file in os.listdir(dataset_dir + "/test"):
        cwd = os.getcwd()
        test_input_names.append(cwd + "/" + dataset_dir + "/test/" + file)
    for file in os.listdir(dataset_dir + "/test_labels"):
        cwd = os.getcwd()
        test_output_names.append(cwd + "/" + dataset_dir + "/test_labels/" + file)
    return train_input_names,train_output_names, train_output_weight_names, val_input_names, val_output_names, test_input_names, test_output_names

##############
# 2 load config
##############
config, unparsed = get_config()
prepare_dirs_and_logger(config)
rng = np.random.RandomState(config.random_seed)
tf.set_random_seed(config.random_seed)

if config.is_train:
    data_path = config.data_path
    batch_size = config.batch_size
    do_shuffle = True
else:
    setattr(config, 'batch_size', 64)
    if config.test_data_path is None:
        data_path = config.data_path
    else:
        data_path = config.test_data_path
    batch_size = config.sample_per_image
    do_shuffle = False
# trainer = Trainer(config) #, data_loader)
if config.is_train:
    save_config(config)
else:
    if not config.load_path:
        raise Exception("[!] You should specify `load_path` to load a pretrained model")
    trainer.test()

##############
# 3 init
##############
tf.reset_default_graph()

#data_loader = data_loader
dataset = config.dataset
beta1 = config.beta1
beta2 = config.beta2
optimizer = config.optimizer
batch_size = config.batch_size
is_edge_weight = config.is_edge_weight

gamma = config.gamma
lambda_k = config.lambda_k

z_num = config.z_num
conv_hidden_num = config.conv_hidden_num
input_scale_size = config.input_scale_size

model_dir = config.model_dir
load_path = config.load_path

use_gpu = config.use_gpu
data_format = config.data_format

#_, height, width, channel = \
#        get_conv_shape(data_loader, data_format)
repeat_num = 4 #int(np.log2(height)) - 2 #lx

start_step = 0
log_step = config.log_step
max_step = config.max_step
save_step = config.save_step
lr_update_step = config.lr_update_step

is_train = config.is_train

step = tf.Variable(0, name='step', trainable=False)

g_lr = tf.Variable(config.g_lr, name='g_lr')

g_lr_update = tf.assign(g_lr, tf.maximum(g_lr * 0.5, config.lr_lower_boundary), name='g_lr_update')

##############
# 4 build Generator
##############
z = tf.placeholder(tf.float32,shape=[None,3,None,None],name='input_img')
gt = tf.placeholder(tf.float32,shape=[None,1,None,None],name='gt_map')
G_logist, G, G_var = Segmentor(z, preset_model='FC-DenseNet56', num_classes=1, data_format=data_format, reuse=False)

g_optimizer = tf.train.RMSPropOptimizer(learning_rate=g_lr, decay=0.9995)

seg_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=G_logist, labels=gt))

g_optim = g_optimizer.minimize(seg_loss, global_step=step, var_list=G_var)

summary_op = tf.summary.merge([
    tf.summary.image("z", denorm_img(z, data_format)),
    tf.summary.image("gt", denorm_img(gt, data_format)),
    tf.summary.image("G", denorm_img(G, data_format)),
    tf.summary.scalar("misc/g_lr", g_lr),
    tf.summary.scalar("loss/seg_loss", seg_loss),
    tf.summary.histogram("lx/G_hist", G),
    tf.summary.histogram("lx/G_logist_hist", G_logist),

])

##############
# 6 session
##############
saver = tf.train.Saver()
summary_writer = tf.summary.FileWriter(model_dir)

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())

sv = tf.train.Supervisor(logdir=model_dir,
                        is_chief=True,
                        saver=saver,
                        summary_op=None,
                        summary_writer=summary_writer,
                        save_model_secs=7200, #checkpoint every 3600 seconds
                        global_step=step,
                        ready_for_local_init_op=None)

gpu_options = tf.GPUOptions(allow_growth=True)
sess_config = tf.ConfigProto(allow_soft_placement=True,
                            gpu_options=gpu_options)
sess = sv.prepare_or_wait_for_session(config=sess_config)

##############
# 7 Load the data
##############
print("Loading the data ...")
train_input_names, train_output_names, train_output_weight_names, val_input_names, val_output_names, test_input_names, test_output_names = prepare_data(config)

print(len(train_input_names),len(train_output_names),len(train_output_weight_names))
print(len(val_input_names),len(val_output_names),len(test_input_names),len(test_output_names))

class_names_list = helpers.get_class_list(os.path.join(dataset, "class_list.txt"))
class_names_string = ""
for class_name in class_names_list:
    if not class_name == class_names_list[-1]:
        class_names_string = class_names_string + class_name + ", "
    else:
        class_names_string = class_names_string + class_name

num_classes = len(class_names_list)

# Which validation images doe we want
val_indices = []
num_vals = min(config.num_val_images, len(val_input_names))
for i in range(num_vals):
    ind = random.randint(0, len(val_input_names) - 1)
    val_indices.append(ind)
id_list = np.random.permutation(len(train_input_names))

##############
# 8 start training
##############
prev_measure = 1
measure_history = deque([0]*lr_update_step, lr_update_step)
for step in trange(start_step, max_step):
    id = id_list[step%len(train_input_names)]
    input_image = np.expand_dims(cv2.cvtColor(cv2.imread(train_input_names[id],-1)[:256,:256,:], cv2.COLOR_BGR2RGB),0) / 127.5 -1
    gt_map = cv2.imread(train_output_names[id],-1)
    gt_map /= np.max(gt_map)
    gt_map = np.expand_dims(np.expand_dims(gt_map,2),0)
    input_image = np.transpose(input_image, [0, 3, 1, 2])
    gt_map = np.transpose(gt_map, [0, 3, 1, 2])
    fetch_dict = {
        "seg_loss": seg_loss,
        "g_optim": g_optim,
    }
    if step % log_step == 0:
        fetch_dict.update({
            "summary": summary_op,
            # "seg_loss": seg_loss,
        })

    #result = sess.run(fetch_dict)
    result = sess.run(fetch_dict,feed_dict={z: input_image, gt:gt_map})

    if step % log_step == 0:
        summary_writer.add_summary(result['summary'], step)
        summary_writer.flush()

    if step % len(train_input_names) == len(train_input_names)-1:
        target=open(model_dir+"/val_scores.txt",'w')
        target.write("val_name, avg_accuracy, precision, recall, f1 score, mean iou\n")
        scores_list = []
        class_scores_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        iou_list = []
        for id in val_indices:
            input_image = np.expand_dims(cv2.cvtColor(cv2.imread(val_input_names[id],-1), cv2.COLOR_BGR2RGB)[:256,:256,:],0)/255.0
            gt_map = np.expand_dims(cv2.imread(val_output_names[id],-1)[:256,:256],2)
            gt_map /= np.max(gt_map)
            input_image = np.transpose(input_image, [0, 3, 1, 2])

            output_image = sess.run(G, {z: input_image})
            output_image = np.clip(output_image[0].transpose([1,2,0]),0,1)
            st = time.time()

            print('output_image.shape, gt.shape:',output_image.shape, gt_map.shape)
            output_image = helpers.reverse_one_hot(output_image)
            out_vis_image = helpers.colour_code_segmentation(output_image)
            accuracy = utils_seg.compute_avg_accuracy(output_image, gt_map)
            class_accuracies = utils_seg.compute_class_accuracies(output_image, gt_map, 2)
            prec = utils_seg.precision(output_image, gt_map)
            rec = utils_seg.recall(output_image, gt_map)
            f1 = utils_seg.f1score(output_image, gt_map)
            iou = utils_seg.compute_mean_iou(output_image, gt_map)

            file_name = utils_seg.filepath_to_name(val_input_names[ind])
            target.write("%s, %f, %f, %f, %f, %f"%(file_name, accuracy, prec, rec, f1, iou))
            for item in class_accuracies:
                target.write(", %f"%(item))
            target.write("\n")

            scores_list.append(accuracy)
            class_scores_list.append(class_accuracies)
            precision_list.append(prec)
            recall_list.append(rec)
            f1_list.append(f1)
            iou_list.append(iou)

            gt_map = helpers.reverse_one_hot(helpers.one_hot_it(gt_map))
            gt_map = helpers.colour_code_segmentation(gt_map)

            file_name = os.path.basename(val_input_names[ind])
            file_name = os.path.splitext(file_name)[0]
            cv2.imwrite(model_dir+"/%s_pred.png"%(ind),np.uint8(out_vis_image))
            cv2.imwrite(model_dir+"/%s_gt.png"%(ind),np.uint8(gt_map))
        avg_score = np.mean(scores_list)
        avg_precision = np.mean(precision_list)
        avg_recall = np.mean(recall_list)
        avg_f1 = np.mean(f1_list)
        avg_iou = np.mean(iou_list)

        print("\nAverage validation accuracy for iteration # %06d = %f"% (step, avg_score))
        print("Average per class validation accuracies for epoch # %06d:"% (step))
#         for index, item in enumerate(class_avg_scores):
#             print("%s = %f" % (class_names_list[index], item))
        print("Validation precision = ", avg_precision)
        print("Validation recall = ", avg_recall)
        print("Validation F1 score = ", avg_f1)
        print("Validation IoU score = ", avg_iou)

    if step % lr_update_step == lr_update_step - 1:
        sess.run([g_lr_update])
        # sess.run([d_lr_update])
        #cur_measure = np.mean(measure_history)
        #if cur_measure > prev_measure * 0.99:
        #prev_measure = cur_measure
