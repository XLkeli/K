import numpy as np
import tensorflow as tf


def glorot(shape, dtype=tf.float32, scope='default'):
    # 定义一个函数glorot，用于初始化权重矩阵，采用Glorot初始化方法
    with tf.variable_scope(scope):   # 使用TensorFlow的变量作用域，将下面的操作放在指定的变量作用域中
        init_range = np.sqrt(6.0 / (shape[0] + shape[1]))    # 使用Glorot初始化方法，计算权重初始化的范围
        init = tf.random_uniform(
            shape, minval=-init_range, maxval=init_range, dtype=dtype)
        # 使用均匀分布在初始化范围内生成随机数，作为权重的初始值
        return tf.Variable(init)     # 将初始化的权重作为TensorFlow变量返回


def zeros(shape, dtype=tf.float32, scope='default'):  # 定义一个函数zeros，用于初始化偏置项，将其设为零
    with tf.variable_scope(scope):   # 同样使用TensorFlow的变量作用域
        init = tf.zeros(shape, dtype=dtype)   # 使用零初始化方法，生成初始化为零的张量
        return tf.Variable(init)   # 将初始化的偏置项作为TensorFlow变量返回


class GraphCNN(object):  # 定义一个类GraphCNN，表示图卷积网络

    def __init__(self, inputs, input_dim, hid_dims, output_dim,
                 max_depth, act_fn, scope='gcn'):

        self.inputs = inputs             # 将输入数据保存到类的实例变量中
        self.input_dim = input_dim       # 将输入数据的维度保存到类的实例变量中
        self.hid_dims = hid_dims
        self.output_dim = output_dim
        self.max_depth = max_depth   # 将图卷积的最大深度保存到类的实例变量中
        self.act_fn = act_fn  # 将激活函数保存到类的实例变量中
        self.scope = scope  # 将变量作用域的名称保存到类的实例变量中

        # initialize message passing transformation parameters
        self.prep_weights, self.prep_bias = \
            self.init(self.input_dim, self.hid_dims, self.output_dim)
        self.proc_weights, self.proc_bias = \
            self.init(self.output_dim, self.hid_dims, self.output_dim)
        self.agg_weights, self.agg_bias = \
            self.init(self.output_dim, self.hid_dims, self.output_dim)
        self.outputs = self.forward()

    def init(self, input_dim, hid_dims, output_dim):
        # Initialize the parameters
        weights = []
        bias = []
        curr_in_dim = input_dim  # 初始化当前输入维度为输入数据的维度

        # Hidden layers
        for hid_dim in hid_dims:
            weights.append(
                glorot([curr_in_dim, hid_dim], scope=self.scope))  # Glorot初始化权重
            bias.append(
                zeros([hid_dim], scope=self.scope))   # 零初始化偏置项
            curr_in_dim = hid_dim  # 更新当前输入维度

        # Output layer
        weights.append(glorot([curr_in_dim, output_dim], scope=self.scope))
        bias.append(zeros([output_dim], scope=self.scope))
        return weights, bias  # 返回初始化得到的权重和偏置项列表

    def forward(self):
        x = self.inputs  # 将输入数据保存到变量x中

        # Raise x into higher dimension
        for l in range(len(self.prep_weights)):
            x = tf.matmul(x, self.prep_weights[l])  # 矩阵相乘
            x += self.prep_bias[l]  # 加上偏置项
            x = self.act_fn(x)  # 应用激活函数

        for d in range(self.max_depth):
            y = x
            # Process the features
            for l in range(len(self.proc_weights)):
                y = tf.matmul(y, self.proc_weights[l])  # 矩阵相乘
                y += self.proc_bias[l]
                y = self.act_fn(y)
            # Aggregate features
            for l in range(len(self.agg_weights)):
                y = tf.matmul(y, self.agg_weights[l])
                y += self.agg_bias[l]
                y = self.act_fn(y)

            # assemble neighboring information
            x = x + y
        return x
