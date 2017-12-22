#!/usr/bin/env python
"""
This is a straightforward Python implementation of a generative adversarial network.
The code is derived from the O'Reilly interactive tutorial on GANs
(https://www.oreilly.com/learning/generative-adversarial-networks-for-beginners).

The tutorial's code trades efficiency for clarity in explaining how GANs function;
this script refactors a few things to improve performance, especially on GPU machines.
In particular, it uses a TensorFlow operation to generate random z values and pass them
to the generator; this way, more computations are contained entirely within the
TensorFlow graph.

A version of this model with explanatory notes is also available on GitHub
at https://github.com/jonbruner/generative-adversarial-networks.

This script requires TensorFlow and its dependencies in order to run. Please see
the readme for guidance on installing TensorFlow.

This script won't print summary statistics in the terminal during training;
track progress and see sample images in TensorBoard.
"""

import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
import numpy as np

# Load MNIST data
from tensorflow.examples.tutorials.mnist import input_data
#读取数据集
mnist = input_data.read_data_sets("MNIST_data/")



# Define the discriminator network
#具有两个卷积层和两个全连接层，特征为5x5
def discriminator(images, reuse_variables=None):
    #遍历各层数据
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables) as scope:
        # First convolutional and pool layers
        # This finds 32 different 5 x 5 pixel features
        #截断正态分布初始化权值
        d_w1 = tf.get_variable('d_w1', [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
        #初始化偏置
        d_b1 = tf.get_variable('d_b1', [32], initializer=tf.constant_initializer(0))
        #标准库中的卷积神经网络
        d1 = tf.nn.conv2d(input=images, filter=d_w1, strides=[1, 1, 1, 1], padding='SAME')
        #增加偏置
        d1 = d1 + d_b1
        #激活函数
        d1 = tf.nn.relu(d1)
        #池化来处理特征
        d1 = tf.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Second convolutional and pool layers
        # This finds 64 different 5 x 5 pixel features
        # 截断正态分布初始化权值
        d_w2 = tf.get_variable('d_w2', [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
        # 初始化偏置
        d_b2 = tf.get_variable('d_b2', [64], initializer=tf.constant_initializer(0))
        # 标准库中的卷积神经网络
        d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 1, 1, 1], padding='SAME')
        # 增加偏置
        d2 = d2 + d_b2
        # 激活函数
        d2 = tf.nn.relu(d2)
        # 池化来处理特征
        d2 = tf.nn.avg_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # First fully connected layer
        # 截断正态分布初始化权值
        d_w3 = tf.get_variable('d_w3', [7 * 7 * 64, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
        # 初始化偏置
        d_b3 = tf.get_variable('d_b3', [1024], initializer=tf.constant_initializer(0))
        #整合数据
        d3 = tf.reshape(d2, [-1, 7 * 7 * 64])
        #矩阵相乘获得函数值
        d3 = tf.matmul(d3, d_w3)
        #增加偏置
        d3 = d3 + d_b3
        #激活函数
        d3 = tf.nn.relu(d3)

        # Second fully connected layer
        #截断正态分布初始化权值
        d_w4 = tf.get_variable('d_w4', [1024, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
        #初始化偏置
        d_b4 = tf.get_variable('d_b4', [1], initializer=tf.constant_initializer(0))
        #与权重相乘，增加偏置
        d4 = tf.matmul(d3, d_w4) + d_b4
        # d4 contains unscaled values
        #返回一个标量，可认为是置信值
        return d4

# Define the generator network
#定义生成网络
def generator(z, batch_size, z_dim):
    #输入为56x56大小，设置权重
    g_w1 = tf.get_variable('g_w1', [z_dim, 3136], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    #设置偏置，大小同上
    g_b1 = tf.get_variable('g_b1', [3136], initializer=tf.truncated_normal_initializer(stddev=0.02))
    #乘上权重+偏置
    g1 = tf.matmul(z, g_w1) + g_b1
    #向量转化为数组
    g1 = tf.reshape(g1, [-1, 56, 56, 1])
    #用于归一化，加快运算速度
    g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='bn1')
    #激活函数
    g1 = tf.nn.relu(g1)

    # Generate 50 features
    #第二层网络权重
    g_w2 = tf.get_variable('g_w2', [3, 3, 1, z_dim/2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    #第二层网络偏置
    g_b2 = tf.get_variable('g_b2', [z_dim/2], initializer=tf.truncated_normal_initializer(stddev=0.02))
    #引用标准网络
    g2 = tf.nn.conv2d(g1, g_w2, strides=[1, 2, 2, 1], padding='SAME')
    #加偏置
    g2 = g2 + g_b2
    #用于归一化，加快运算速度
    g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='bn2')
    #激活函数
    g2 = tf.nn.relu(g2)
    #调整大小
    g2 = tf.image.resize_images(g2, [56, 56])

    # Generate 25 features
    #第三层网络权重
    g_w3 = tf.get_variable('g_w3', [3, 3, z_dim/2, z_dim/4], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    #第三层网络偏置
    g_b3 = tf.get_variable('g_b3', [z_dim/4], initializer=tf.truncated_normal_initializer(stddev=0.02))
    #引用标准网络
    g3 = tf.nn.conv2d(g2, g_w3, strides=[1, 2, 2, 1], padding='SAME')
    #加偏置
    g3 = g3 + g_b3
    # 用于归一化，加快运算速度
    g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='bn3')
    #激活函数
    g3 = tf.nn.relu(g3)
    #调整大小
    g3 = tf.image.resize_images(g3, [56, 56])

    # Final convolution with one output channel
    #第四层权重
    g_w4 = tf.get_variable('g_w4', [1, 1, z_dim/4, 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    #第四层偏置
    g_b4 = tf.get_variable('g_b4', [1], initializer=tf.truncated_normal_initializer(stddev=0.02))
    #标准网络
    g4 = tf.nn.conv2d(g3, g_w4, strides=[1, 2, 2, 1], padding='SAME')
    #加偏置
    g4 = g4 + g_b4
    #激活函数使得图像变为灰度图像
    g4 = tf.sigmoid(g4)
    # Dimensions of g4: batch_size x 28 x 28 x 1
    #返回28x28图像数据
    return g4


#批个数定义
z_dimensions = 100

#设定维度大小
z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions])

#生成器输出
generated_image_output = generator(z_placeholder, 1, z_dimensions)
#生成噪音
z_batch = np.random.normal(0, 1, [1, z_dimensions])

with tf.Session() as sess:
    #初始化变量
    sess.run(tf.global_variables_initializer())
    #生成图片
    generated_image = sess.run(generated_image_output,
                                feed_dict={z_placeholder: z_batch})
    #生成的向量重塑
    generated_image = generated_image.reshape([28, 28])
    plt.imshow(generated_image, cmap='Greys')

#设定张量图
tf.reset_default_graph()
#设定包大小
batch_size = 50
#生成器噪音
z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions], name='z_placeholder') 
# z_placeholder is for feeding input noise to the generator
#判别器输入
x_placeholder = tf.placeholder(tf.float32, shape = [None,28,28,1], name='x_placeholder') 
# x_placeholder is for feeding input images to the discriminator
#存储生成器图片
Gz = generator(z_placeholder, batch_size, z_dimensions) 
# Gz holds the generated images
#存储判别器的输出
Dx = discriminator(x_placeholder) 
# Dx will hold discriminator prediction probabilities
# for the real MNIST images
#存储生成器图片的判别后的概率
Dg = discriminator(Gz, reuse_variables=True)
# Dg will hold discriminator prediction probabilities for generated images


#判别器真实图片输出与1的交叉熵
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dx, labels = tf.ones_like(Dx)))
#判别器生成图片输出与0的交叉熵
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.zeros_like(Dg)))


g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.ones_like(Dg)))  

#存储训练变量
tvars = tf.trainable_variables()
#判别器权重、偏置
d_vars = [var for var in tvars if 'd_' in var.name]
#生成器权重、偏置
g_vars = [var for var in tvars if 'g_' in var.name]
#输出判别器name
print([v.name for v in d_vars])
#输出生成器name
print([v.name for v in g_vars])


# Train the discriminator
#训练判别器（生成图片）
d_trainer_fake = tf.train.AdamOptimizer(0.0003).minimize(d_loss_fake, var_list=d_vars)
#训练判别器（真实图片）
d_trainer_real = tf.train.AdamOptimizer(0.0003).minimize(d_loss_real, var_list=d_vars)

# Train the generator
#训练生成器
g_trainer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_vars)


# From this point forward, reuse variables
#获取变量
tf.get_variable_scope().reuse_variables()


#生成器损失
tf.summary.scalar('Generator_loss', g_loss)
#判别器对真实图片的损失
tf.summary.scalar('Discriminator_loss_real', d_loss_real)
#判别器对生成图片的损失
tf.summary.scalar('Discriminator_loss_fake', d_loss_fake)
#获取生成图
images_for_tensorboard = generator(z_placeholder, batch_size, z_dimensions)
#图像写入summary便于查看
tf.summary.image('Generated_images', images_for_tensorboard, 5)
#管理summary
merged = tf.summary.merge_all()
#图像名设定
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
#结果写入tensorboard
writer = tf.summary.FileWriter(logdir, sess.graph)

#'''
#设定会话
sess = tf.Session()
#启动会话
sess.run(tf.global_variables_initializer())

# Pre-train discriminator
#预训练，便于设定权重和偏置
for i in range(300):
    #初始化zbatch用于预训练
    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
    #真实图片的遍历
    real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
    #获取真实图片损失和生成图片损失
    _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                           {x_placeholder: real_image_batch, z_placeholder: z_batch})
    #每100批进行一次输出
    if(i % 100 == 0):
        print("dLossReal:", dLossReal, "dLossFake:", dLossFake)

# Train generator and discriminator together
#同时进行生成器和判别器的训练
for i in range(100000):
    #提取真实图片批
    real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
    #初始化zbatch用于预训练
    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])

    # Train discriminator on both real and fake images
    #获取真实图片损失和生成图片损失
    _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                           {x_placeholder: real_image_batch, z_placeholder: z_batch})

    # Train generator
    #初始化zbatch用于预训练
    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
    #启动会话训练
    _ = sess.run(g_trainer, feed_dict={z_placeholder: z_batch})
    #每10次更新一次
    if i % 10 == 0:
        # Update TensorBoard with summary statistics
        #zbatch初始化
        z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
        #赋值summary便于管理
        summary = sess.run(merged, {z_placeholder: z_batch, x_placeholder: real_image_batch})
        #写入summary
        writer.add_summary(summary, i)
    #每100次输出一次
    if i % 100 == 0:
        # Every 100 iterations, show a generated image
        #输出遍历次数、时间
        print("Iteration:", i, "at", datetime.datetime.now())
        #初始化zbatch用于预训练
        z_batch = np.random.normal(0, 1, size=[1, z_dimensions])
        #生成图像
        generated_images = generator(z_placeholder, 1, z_dimensions)
        #启动会话
        images = sess.run(generated_images, {z_placeholder: z_batch})
        #图像组织
        plt.imshow(images[0].reshape([28, 28]), cmap='Greys')
        #图像显示
        plt.show()

        # Show discriminator's estimate
        #组织图像
        im = images[0].reshape([1, 28, 28, 1])
        #图像判别
        result = discriminator(x_placeholder)
        #估计结果
        estimate = sess.run(result, {x_placeholder: im})
        #输出结果
        print("Estimate:", estimate)
#'''
#调用存储器
saver = tf.train.Saver()
#开始会话
with tf.Session() as sess:
    #获取存储的模型
    saver.restore(sess, 'pretrained-model/pretrained_gan.ckpt')
    #设定批大小
    z_batch = np.random.normal(0, 1, size=[10, z_dimensions])
    #存储模型
    z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions], name='z_placeholder')
    #运用已有模型生成图像
    generated_images = generator(z_placeholder, 10, z_dimensions)
    #启动会话
    images = sess.run(generated_images, {z_placeholder: z_batch})
    #遍历输出图像
    for i in range(10):
        #提取灰度图像
        plt.imshow(images[i].reshape([28, 28]), cmap='Greys')
        #图像显示
        plt.show()


