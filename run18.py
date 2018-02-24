#coding:utf-8

##############
# 1 load module
##############

from __future__ import print_function
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='3'

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
##############
# 4 build Generator
##############
z = tf.placeholder(tf.float32,shape=[None,3,None,None],name='input_img')
gt = tf.placeholder(tf.float32,shape=[None,1,None,None],name='gt_map')
G_logist, G, G_var = Segmentor(z, preset_model='FC-DenseNet56', num_classes=1, data_format=data_format, reuse=False)
# print(G_var)

# use_pretrain = True
# checkpoints_file = '0041/model.ckpt'

# from tensorflow.python import pywrap_tensorflow
# reader = pywrap_tensorflow.NewCheckpointReader(checkpoints_file)
# var_to_shape_map = reader.get_variable_to_shape_map()
# # for key in var_to_shape_map:
# #     print("tensor_name: ", key)
# with tf.Session() as sess_init:
#    variables_to_restore = slim.get_variables_to_restore()
#    restorer = tf.train.Saver(variables_to_restore)
#    restorer.restore(sess_init, checkpoints_file)
   # assign_op, feed_dict_init = slim.assign_from_values({
   #      'FC-DenseNet56/logits/weights' : reader.get_tensor('FC-DenseNet56/logits/weights')[:,:,:,1:],
   #  })
   # assign_op, feed_dict_init = slim.assign_from_values({
   #      'FC-DenseNet56/logits/biases' : reader.get_tensor('FC-DenseNet56/logits/biases')[1],
   #  })
   # sess_init.run(assign_op, feed_dict_init)

k_t = tf.Variable(0., trainable=False, name='k_t')
step = tf.Variable(0, name='step', trainable=False)

g_lr = tf.Variable(config.g_lr, name='g_lr')
d_lr = tf.Variable(config.d_lr, name='d_lr')

g_lr_update = tf.assign(g_lr, tf.maximum(g_lr * 0.5, config.lr_lower_boundary), name='g_lr_update')
d_lr_update = tf.assign(d_lr, tf.maximum(d_lr * 0.5, config.lr_lower_boundary), name='d_lr_update')

##############
# 5. build Discriminator
##############
d_out, GG, xx, D_var = Discriminator_Product_small(z, gt, G, z_num, repeat_num, conv_hidden_num, data_format)
AE_G, AE_x = tf.split(d_out, 2)

if optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer
else:
    raise Exception("[!] Caution! Paper didn't use {} opimizer other than Adam".format(config.optimizer))

g_optimizer, d_optimizer = optimizer(g_lr), optimizer(d_lr)

seg_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=G_logist, labels=gt))
seg_l1_loss = tf.reduce_mean(tf.abs(G - gt))

d_loss_real = (0.01+tf.reduce_sum(gt*tf.abs(AE_x - xx)))/(tf.reduce_sum(gt)+0.01)
d_loss_fake = (0.01+tf.reduce_sum(gt*tf.abs(AE_G - GG)))/(tf.reduce_sum(gt)+0.01)

g_loss = (1-config.seg_weight)*d_loss_fake + config.seg_weight*seg_loss
d_loss = d_loss_real - k_t * g_loss

# g_loss = seg_loss

d_optim = d_optimizer.minimize(d_loss, var_list=D_var)
g_optim = g_optimizer.minimize(g_loss, global_step=step, var_list=G_var)

balance = gamma * d_loss_real - g_loss
measure = d_loss_real + tf.abs(balance)

with tf.control_dependencies([d_optim, g_optim]):
    k_update = tf.assign(
        k_t, tf.clip_by_value(k_t + lambda_k * balance, 0, 1))

summary_op = tf.summary.merge([
    tf.summary.image("z", denorm_img(z, data_format)),
    tf.summary.image("gt", denorm_img(gt, data_format)),
    tf.summary.image("G", denorm_img(G, data_format)),
    tf.summary.image("GG", denorm_img(GG, data_format)),
    tf.summary.image("xx", denorm_img(xx, data_format)),
    tf.summary.image("AE_G", denorm_img(AE_G, data_format)),
    tf.summary.image("AE_x", denorm_img(AE_x, data_format)),
    tf.summary.scalar("loss/d_loss", d_loss),
    tf.summary.scalar("loss/d_loss_real", d_loss_real),
    tf.summary.scalar("loss/d_loss_fake", d_loss_fake),
    tf.summary.scalar("loss/g_loss", g_loss),
    tf.summary.scalar("loss/seg_loss", seg_loss),
    tf.summary.scalar("loss/seg_l1_loss", seg_l1_loss),
    tf.summary.scalar("misc/measure", measure),
    tf.summary.scalar("misc/k_t", k_t),
    tf.summary.scalar("misc/d_lr", d_lr),
    tf.summary.scalar("misc/g_lr", g_lr),
    tf.summary.scalar("misc/balance", balance),
    tf.summary.histogram("lx/G_hist", G),
    tf.summary.histogram("lx/G_logist_hist", G_logist),
    tf.summary.histogram("lx/GG_hist", GG),
    tf.summary.histogram("lx/xx_hist", xx),
    tf.summary.histogram("lx/AE_G_hist", AE_G),
    tf.summary.histogram("lx/AE_xx_hist", AE_x),
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
    input_image = np.expand_dims(cv2.cvtColor(cv2.imread(train_input_names[id],-1)[:256,:256,:], cv2.COLOR_BGR2RGB),0) / 255.0
    gt_map = cv2.imread(train_output_names[id],-1)
    gt_map /= np.max(gt_map)
    # print('gt_map.shape:',gt_map.shape, np.unique(gt_map))
    # gt_map = np.expand_dims(helpers.one_hot_it(gt_map,num_classes=2),0)
    gt_map = np.expand_dims(np.expand_dims(gt_map,2),0)
    # input_image = np.expand_dims(cv2.resize(cv2.cvtColor(cv2.imread(train_input_names[id],-1), cv2.COLOR_BGR2RGB),(256,256)),0)/127.5 - 1.
    # gt_map = np.expand_dims(np.expand_dims(cv2.resize(cv2.imread(train_output_names[id],-1),(256,256)),0),3)
    input_image = np.transpose(input_image, [0, 3, 1, 2])
    gt_map = np.transpose(gt_map, [0, 3, 1, 2])
    fetch_dict = {
        "k_update": k_update,
        "measure": measure,
    }
    if step % log_step == 0:
        fetch_dict.update({
            "summary": summary_op,
            "g_loss": g_loss,
            "d_loss": d_loss,
            "k_t": k_t,
        })

    #result = sess.run(fetch_dict)
    result = sess.run(fetch_dict,feed_dict={z: input_image, gt:gt_map})

    measure_cur = result['measure']
    measure_history.append(measure_cur)

    if step % log_step == 0:
        summary_writer.add_summary(result['summary'], step)
        summary_writer.flush()

        g_loss_cur = result['g_loss']
        d_loss_cur = result['d_loss']
        k_t_cur = result['k_t']

        print("[{}/{}] Loss_D: {:.6f} Loss_G: {:.6f} measure: {:.4f}, k_t: {:.4f}". \
              format(step, max_step, d_loss_cur, g_loss_cur, measure_cur, k_t_cur))

    if step % (log_step * 10) == 0:
        print('generate')
        x_fake = generate(sess, input_image, model_dir, idx=step)
        x_fake = input_image*x_fake
        x_fixed = input_image*gt_map
        print(x_fixed.shape,x_fake.shape)
        autoencode(sess, input_image, x_fixed, model_dir, idx=step, x_fake=x_fake)

    if step % 2000 == 2000-1:
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
            # input_image = np.expand_dims(cv2.resize(cv2.cvtColor(cv2.imread(val_input_names[id],-1), cv2.COLOR_BGR2RGB),(256,256)),0)/127.5 - 1.
            # gt_map = np.expand_dims(cv2.resize(cv2.imread(val_output_names[id],-1),(256,256)),2)
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
        sess.run([g_lr_update, d_lr_update])
        # sess.run([d_lr_update])
        #cur_measure = np.mean(measure_history)
        #if cur_measure > prev_measure * 0.99:
        #prev_measure = cur_measure
