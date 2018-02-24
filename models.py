import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
from FC_DenseNet_Tiramisu import build_fc_densenet

def GeneratorCNN(z, hidden_num, output_num, repeat_num, data_format, reuse):
    with tf.variable_scope("G", reuse=reuse) as vs:
        num_output = int(np.prod([8, 8, hidden_num]))
        x = slim.fully_connected(z, num_output, activation_fn=None)
        x = reshape(x, 8, 8, hidden_num, data_format)
        
        for idx in range(repeat_num):
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            if idx < repeat_num - 1:
                x = upscale(x, 2, data_format)

        out = slim.conv2d(x, 3, 3, 1, activation_fn=None, data_format=data_format)

    variables = tf.contrib.framework.get_variables(vs)
    return out, variables

def Segmentor(z, preset_model, num_classes, data_format, reuse): #here z denotes input image
    with tf.variable_scope(preset_model, reuse=reuse) as vs:
        logist, prob, variables = build_fc_densenet(z, preset_model = preset_model, num_classes=num_classes, is_bottneck=False, compression_rate=1, data_format=data_format)
    variables = tf.contrib.framework.get_variables(vs)
    return logist, prob, variables

def DiscriminatorCNN(x, input_channel, z_num, repeat_num, hidden_num, data_format):
    with tf.variable_scope("D") as vs:
        # Encoder
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)

        prev_channel_num = hidden_num
        for idx in range(repeat_num):
            channel_num = hidden_num * (idx + 1)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            if idx < repeat_num - 1:
                x = slim.conv2d(x, channel_num, 3, 2, activation_fn=tf.nn.elu, data_format=data_format)
                #x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID')
        
        x = tf.reshape(x, [-1, np.prod([8, 8, channel_num])])
        z = x = slim.fully_connected(x, z_num, activation_fn=None)

        # Decoder
        num_output = int(np.prod([8, 8, hidden_num]))
        x = slim.fully_connected(x, num_output, activation_fn=None)
        x = reshape(x, 8, 8, hidden_num, data_format)
        
        for idx in range(repeat_num):
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            if idx < repeat_num - 1:
                x = upscale(x, 2, data_format)
                
        out = slim.conv2d(x, input_channel, 3, 1, activation_fn=None, data_format=data_format)

    variables = tf.contrib.framework.get_variables(vs)
    return out, z, variables

def Discriminator(z, gt, G, z_num, repeat_num, hidden_num, data_format): 
    with tf.variable_scope("D") as vs:
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                      data_format=data_format,
                      activation_fn=tf.nn.elu,
                      #weights_initializer=tf.truncated_normal_initializer(0, 0.05),
                      weights_regularizer=slim.l2_regularizer(0.0005)
                           ):
            z = slim.conv2d(z, 16, 3)
            z = slim.conv2d(z, 64, 3)
            G = slim.conv2d(G, 64, 3)
            gt = slim.conv2d(gt, 64, 3)
            GG = tf.concat([z, G], 1)
            xx = tf.concat([z, gt], 1)
            print('GG.shape','xx.shape',GG.shape,xx.shape)
            x = tf.concat([GG,xx],0)
            # Encoder
            # x = slim.conv2d(x, hidden_num, 3)

            prev_channel_num = hidden_num
            for idx in range(repeat_num):
                channel_num = hidden_num * (idx + 1)
                x = slim.conv2d(x, channel_num, 3)
                # x = slim.dropout(x, keep_prob=0.5)
                x = slim.conv2d(x, channel_num, 3)
                # x = slim.dropout(x, keep_prob=0.5)
                if idx < repeat_num - 1: #0-3
                    x = slim.conv2d(x, channel_num, 3, 2)
                    # x = slim.dropout(x, keep_prob=0.5)
                    # x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID')
                    # x = slim.pool(x, [2, 2], stride=[2, 2], pooling_type='MAX', data_format='NCHW')
            down_size = 256/pow(2,repeat_num-1)
            x = tf.reshape(x, [-1, np.prod([down_size, down_size, channel_num])]) #8*8*(64*5)<---128/16,128/16,(64*5)
            z = x = slim.fully_connected(x, z_num, activation_fn=None)

            # Decoder
            num_output = int(np.prod([down_size, down_size, hidden_num]))
            x = slim.fully_connected(x, num_output, activation_fn=None)
            x = reshape(x, down_size, down_size, hidden_num, data_format)

            for idx in range(repeat_num):
                x = slim.conv2d(x, hidden_num, 3, 1)
                # x = slim.dropout(x, keep_prob=0.5)
                x = slim.conv2d(x, hidden_num, 3, 1)
                # x = slim.dropout(x, keep_prob=0.5)
                if idx < repeat_num - 1:
                    x = upscale(x, 2, data_format)
                    # x = slim.dropout(x, keep_prob=0.5)
                    # x = slim.conv2d_transpose(x, hidden_num, 3, 2)

            out = slim.conv2d(x, 128, 3)

    variables = tf.contrib.framework.get_variables(vs)
    return out, GG, xx, variables

def Discriminator_small(z, gt, G, z_num, repeat_num, hidden_num, data_format): 
    with tf.variable_scope("D") as vs:
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                      data_format=data_format,
                      # activation_fn=tf.nn.elu,
                      #weights_initializer=tf.truncated_normal_initializer(0, 0.05),
                      # weights_regularizer=slim.l2_regularizer(0.0005)
                           ):
            # z = slim.conv2d(z, 16, 3)
            # z = slim.conv2d(z, 64, 3)
            # G = slim.conv2d(G, 16, 3)
            # gt = slim.conv2d(gt, 64, 3)
            GG = tf.concat([z, G], 1)
            xx = tf.concat([z, gt], 1)
            print('GG.shape','xx.shape',GG.shape,xx.shape)
            x = tf.concat([GG,xx],0)
            # Encoder-Decoder
            x = slim.conv2d(x, hidden_num, 3, 2)
            x = slim.conv2d(x, hidden_num, 3, 2)
            x = slim.conv2d(x, hidden_num, 3, 2)

            x = slim.conv2d_transpose(x, hidden_num, 3, 2)
            x = slim.conv2d_transpose(x, hidden_num, 3, 2)
            x = slim.conv2d_transpose(x, hidden_num, 3, 2)

            out = slim.conv2d(x, 5, 3)

    variables = tf.contrib.framework.get_variables(vs)
    return out, GG, xx, variables

def Discriminator_Product(z, gt, G, z_num, repeat_num, hidden_num, data_format): 
    with tf.variable_scope("D") as vs:
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                      data_format=data_format,
                      # activation_fn=tf.nn.elu,
                      #weights_initializer=tf.truncated_normal_initializer(0, 0.05),
                      weights_regularizer=slim.l2_regularizer(0.0005)
                           ):
            GG = tf.multiply(z, G)
            xx = tf.multiply(z, gt)
            print('GG.shape','xx.shape',GG.shape,xx.shape)
            x = tf.concat([GG,xx],0)
            # Encoder
            # x = slim.conv2d(x, hidden_num, 3)

            prev_channel_num = hidden_num
            for idx in range(repeat_num):
                channel_num = hidden_num * (idx + 1)
                x = slim.conv2d(x, channel_num, 3)
                # x = slim.dropout(x, keep_prob=0.5)
                x = slim.conv2d(x, channel_num, 3)
                # x = slim.dropout(x, keep_prob=0.5)
                if idx < repeat_num - 1: #0-3
                    x = slim.conv2d(x, channel_num, 3, 2)
                    # x = slim.dropout(x, keep_prob=0.5)
                    # x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID')
                    # x = slim.pool(x, [2, 2], stride=[2, 2], pooling_type='MAX', data_format='NCHW')
            down_size = 256/pow(2,repeat_num-1)
            x = tf.reshape(x, [-1, np.prod([down_size, down_size, channel_num])]) #8*8*(64*5)<---128/16,128/16,(64*5)
            z = x = slim.fully_connected(x, z_num, activation_fn=None)

            # Decoder
            num_output = int(np.prod([down_size, down_size, hidden_num]))
            x = slim.fully_connected(x, num_output, activation_fn=None)
            x = reshape(x, down_size, down_size, hidden_num, data_format)

            for idx in range(repeat_num):
                x = slim.conv2d(x, hidden_num, 3, 1)
                # x = slim.dropout(x, keep_prob=0.5)
                x = slim.conv2d(x, hidden_num, 3, 1)
                # x = slim.dropout(x, keep_prob=0.5)
                if idx < repeat_num - 1:
                    x = upscale(x, 2, data_format)
                    # x = slim.dropout(x, keep_prob=0.5)
                    # x = slim.conv2d_transpose(x, hidden_num, 3, 2)

            out = slim.conv2d(x, 3, 3)

    variables = tf.contrib.framework.get_variables(vs)
    return out, GG, xx, variables

def Discriminator_Product_small(z, gt, G, z_num, repeat_num, hidden_num, data_format): 
    with tf.variable_scope("D") as vs:
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                      data_format=data_format,
                      # activation_fn=tf.nn.elu,
                      #weights_initializer=tf.truncated_normal_initializer(0, 0.05),
                      weights_regularizer=slim.l2_regularizer(0.0005)
                           ):
            GG = tf.multiply(z, G)
            xx = tf.multiply(z, gt)
            print('GG.shape','xx.shape',GG.shape,xx.shape)
            x = tf.concat([GG,xx],0)

            # Encoder-Decoder
            x = slim.conv2d(x, hidden_num, 3, 2)
            x = slim.conv2d(x, hidden_num, 3, 2)
            x = slim.conv2d(x, hidden_num, 3, 2)

            x = slim.conv2d_transpose(x, hidden_num, 3, 2)
            x = slim.conv2d_transpose(x, hidden_num, 3, 2)
            x = slim.conv2d_transpose(x, hidden_num, 3, 2)

            out = slim.conv2d(x, 3, 3)

    variables = tf.contrib.framework.get_variables(vs)
    return out, GG, xx, variables

def Discriminator2(x, input_channel, z_num, repeat_num, hidden_num, data_format): 
    with tf.variable_scope("D") as vs:
        # Encoder
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)

        prev_channel_num = hidden_num
        for idx in range(repeat_num): #log(size)-2=5
            channel_num = hidden_num * (idx + 1)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            if idx < repeat_num - 1: #0-3
                x = slim.conv2d(x, channel_num, 3, 2, activation_fn=tf.nn.elu, data_format=data_format)
                #x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID')
        x = tf.reshape(x, [-1, np.prod([8, 8, channel_num])]) #8*8*(64*5)<---128/16,128/16,(64*5)
        x = slim.fully_connected(x, 1024, activation_fn=None)
        x = slim.fully_connected(x, 1024, activation_fn=None)
        out = slim.fully_connected(x, 2, activation_fn=None)

    variables = tf.contrib.framework.get_variables(vs)
    return out, variables

def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]

def get_conv_shape(tensor, data_format):
    shape = int_shape(tensor)
    # always return [N, H, W, C]
    if data_format == 'NCHW':
        return [shape[0], shape[2], shape[3], shape[1]]
    elif data_format == 'NHWC':
        return shape

def nchw_to_nhwc(x):
    return tf.transpose(x, [0, 2, 3, 1])

def nhwc_to_nchw(x):
    return tf.transpose(x, [0, 3, 1, 2])

def reshape(x, h, w, c, data_format):
    if data_format == 'NCHW':
        x = tf.reshape(x, [-1, c, h, w])
    else:
        x = tf.reshape(x, [-1, h, w, c])
    return x

def resize_nearest_neighbor(x, new_size, data_format):
    if data_format == 'NCHW':
        x = nchw_to_nhwc(x)
        x = tf.image.resize_nearest_neighbor(x, new_size)
        x = nhwc_to_nchw(x)
    else:
        x = tf.image.resize_nearest_neighbor(x, new_size)
    return x

def upscale(x, scale, data_format):
    _, h, w, _ = get_conv_shape(x, data_format)
    return resize_nearest_neighbor(x, (h*scale, w*scale), data_format)
