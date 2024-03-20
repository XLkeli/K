import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tl
import bisect
from .gcn import GraphCNN

# 从自定义的 algorithm.gsn 模块中导入 GraphSNN 类。
# 这个类可能是一个实现图自注意力网络（Graph Self-Attention Network，GSN）的自定义模块。
from .gsn import GraphSNN
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


def discount(x, gamma):
    out = np.zeros(x.shape)
    out[-1] = x[-1]
    for i in reversed(range(len(x) - 1)):
        out[i] = x[i] + gamma * out[i + 1]
    return out


def invoke_model(orchestrate_agent, obs, exp):
    # 定义一个函数，用于调用模型

    # 调用 orchestrate_agent 模型的 invoke_model 方法，获取节点动作、集群动作及其概率，以及节点和集群的输入
    node_act, cluster_act, node_act_probs, cluster_act_probs, node_inputs, cluster_inputs = \
        orchestrate_agent.invoke_model(obs)

    # 将节点动作和集群动作转换为可读的选择列表
    node_choice = [x for x in node_act[0]]
    server_choice = []            # 创建一个空列表，用于存储服务器选择
    for x in cluster_act[0][0]:   # 遍历集群动作的列表
        if x >= 12:
            server_choice.append(x - 11)  # 将经过调整的动作值添加到服务器选择列表
        else:
            server_choice.append(x - 12)

    # 创建与节点动作概率形状相同的全一数组
    node_act_vec = np.ones(node_act_probs.shape)
    # 为存储集群动作概率而创建全一数组
    cluster_act_vec = np.ones(cluster_act_probs.shape)

    # 存储经验信息
    exp['node_inputs'].append(node_inputs)          # 节点输入
    exp['cluster_inputs'].append(cluster_inputs)    # 集群输入
    exp['node_act_vec'].append(node_act_vec)        # 节点动作概率
    exp['cluster_act_vec'].append(cluster_act_vec)  # 集群动作概率

    # 返回节点选择、服务器选择和经验信息
    return node_choice, server_choice, exp


def act_offload_agent(orchestrate_agent, exp, done_tasks, undone_tasks, curr_tasks_in_queue, deploy_state):
    # 定义一个函数，用于执行离线任务

    # 将观察到的状态组成一个观察向量
    obs = [done_tasks, undone_tasks, curr_tasks_in_queue, deploy_state]

    # 调用 invoke_model 函数，获取节点选择、使用执行和经验信息
    node, use_exec, exp = invoke_model(orchestrate_agent, obs, exp)

    # 返回节点选择、使用执行和经验信息
    return node, use_exec, exp


def get_piecewise_linear_fit_baseline(all_cum_rewards, all_wall_time):
    # 定义一个函数，用于获取分段线性拟合的基线

    assert len(all_cum_rewards) == len(all_wall_time)
    # All time
    unique_wall_time = np.unique(np.hstack(all_wall_time))
    # Find baseline value
    baseline_values = {}
    for t in unique_wall_time:
        baseline = 0
        for i in range(len(all_wall_time)):
            # 使用二分查找在当前时间点 t 的位置
            idx = bisect.bisect_left(all_wall_time[i], t)
            if idx == 0:
                baseline += all_cum_rewards[i][idx]
            elif idx == len(all_cum_rewards[i]):
                baseline += all_cum_rewards[i][-1]
            elif all_wall_time[i][idx] == t:
                baseline += all_cum_rewards[i][idx]
            else:
                # 线性插值
                baseline += \
                    (all_cum_rewards[i][idx] - all_cum_rewards[i][idx - 1]) / \
                    (all_wall_time[i][idx] - all_wall_time[i][idx - 1]) * \
                    (t - all_wall_time[i][idx]) + all_cum_rewards[i][idx]

        baseline_values[t] = baseline / float(len(all_wall_time))
    # Output n baselines  输出 n 个基线
    baselines = []
    for wall_time in all_wall_time:
        baseline = np.array([baseline_values[t] for t in wall_time])
        baselines.append(baseline)
    return baselines


def compute_orchestrate_loss(orchestrate_agent, exp, batch_adv, entropy_weight):
    # 定义一个函数，用于计算 orchestrate_agent 的损失
    batch_points = 2   # 批处理的数据点数量
    loss = 0           # 初始化损失为0
    for b in range(batch_points - 1):   # 循环遍历每个数据点，这里实际上只循环一次，因为 batch_points - 1 等于 1
        ba_start = 0  # 定义起始索引为0
        ba_end = -1   # 定义结束索引为-1，表示取所有元素
        # Use a piece of experience
        node_inputs = exp['node_inputs']
        cluster_inputs = exp['cluster_inputs']
        node_act_vec = exp['node_act_vec']
        cluster_act_vec = exp['cluster_act_vec']
        adv = batch_adv[ba_start: ba_end, :]

        # 调用 orchestrate_agent 的 update_gradients 函数更新梯度
        loss = orchestrate_agent.update_gradients(
            node_inputs, cluster_inputs, node_act_vec, cluster_act_vec, adv, entropy_weight)
    return loss


def decrease_var(var, min_var, decay_rate):
    # 定义一个函数，用于递减变量的值
    if var - decay_rate >= min_var:
        # 如果递减后的值仍然大于等于最小值，则递减变量的值
        var -= decay_rate
    else:
        # 如果递减后的值小于最小值，则将变量的值设为最小值
        var = min_var
    return var


def train_orchestrate_agent(orchestrate_agent, exp, entropy_weight, entropy_weight_min, entropy_weight_decay):
    # 定义一个函数，用于训练 orchestrate_agent

    all_cum_reward = []  # 存储所有累积奖励
    all_rewards = exp['reward']  # 获取经验中的奖励数组
    batch_time = exp['wall_time']  # 获取经验中的时间数组
    all_times = batch_time[1:]  # 获取所有时间（除了第一个时间点）
    all_diff_times = np.array(batch_time[1:]) - np.array(batch_time[:-1])  # 计算相邻时间点的时间差
    rewards = np.array([r for (r, t) in zip(all_rewards, all_diff_times)])  # 计算每个时间点的奖励
    cum_reward = discount(rewards, 1)  # 计算累积奖励
    all_cum_reward.append(cum_reward)  # 将累积奖励添加到列表中

    # 计算基线
    baselines = get_piecewise_linear_fit_baseline(all_cum_reward, [all_times])

    # 计算优势值
    batch_adv = all_cum_reward[0] - baselines[0]
    batch_adv = np.reshape(batch_adv, [len(batch_adv), 1])

    # 计算梯度和损失
    loss = compute_orchestrate_loss(
        orchestrate_agent, exp, batch_adv, entropy_weight)

    # 更新熵权重
    entropy_weight = decrease_var(entropy_weight,
                                  entropy_weight_min, entropy_weight_decay)

    #print(loss)
    # 返回熵权重和损失
    return entropy_weight, loss


class Agent(object):
    def __init__(self):
        pass


def expand_act_on_state(state, sub_acts):
    # 定义一个函数，用于在给定状态上扩展动作

    # Expand the state
    # 获取批次大小、节点数量和特征数量
    batch_size = tf.shape(state)[0]   # 使用 TensorFlow 的 tf.shape 函数获取状态张量的第一个维度大小，即批次大小。
    num_nodes = tf.shape(state)[1]    # 使用 TensorFlow 的 tf.shape 函数获取状态张量的第二个维度大小，即节点数量。
    num_features = state.shape[2].value   # 获取状态张量的第三个维度大小，即特征数量。这里使用 value 属性获取维度的静态大小。
    # 获取需要扩展的维度数量
    expand_dim = len(sub_acts)    # 获取需要扩展的维度数量，即给定动作列表 sub_acts 的长度。

    # Replicate the state
    # 使用 TensorFlow 的 tf.tile 函数，将状态张量沿第三个维度（特征维度）进行复制，
    # 复制的次数由 expand_dim 决定。这样做相当于在特征维度上进行复制，以扩展动作。
    state = tf.tile(state, [1, 1, expand_dim])

    state = tf.reshape(state,
                       [batch_size, num_nodes * expand_dim, num_features])

    # Prepare the appended actions
    sub_acts = tf.constant(sub_acts, dtype=tf.float32)
    sub_acts = tf.reshape(sub_acts, [1, 1, expand_dim])
    sub_acts = tf.tile(sub_acts, [1, 1, num_nodes])
    sub_acts = tf.reshape(sub_acts, [1, num_nodes * expand_dim, 1])
    sub_acts = tf.tile(sub_acts, [batch_size, 1, 1])

    # Concatenate expanded state with sub-action features
    concat_state = tf.concat([state, sub_acts], axis=2)

    return concat_state


def leaky_relu(features, alpha=0.2, name=None):
    # 使用 TensorFlow 的 name_scope 定义一个命名范围，以便更好地组织图中的节点
    with ops.name_scope(name, "LeakyRelu", [features, alpha]):
        # 将输入 features 转换为 TensorFlow 张量
        features = ops.convert_to_tensor(features, name="features")
        # 将 alpha 参数转换为 TensorFlow 张量
        alpha = ops.convert_to_tensor(alpha, name="alpha")
        return math_ops.maximum(alpha * features, features)


class OrchestrateAgent(Agent):   # 定义 OrchestrateAgent 类，继承自 Agent 类

    # 整个构造函数的目的是接受一系列参数，并用这些参数初始化 OrchestrateAgent 类的实例。
    # 在深度强化学习中，这样的类通常代表了一个智能体，具有一些状态、动作和策略等成员属性，
    # 同时还包括了一些方法用于智能体的决策和学习。
    def __init__(self, sess, node_input_dim, cluster_input_dim, hid_dims, output_dim,
                 max_depth, executor_levels, eps=1e-6, act_fn=leaky_relu,
                 optimizer=tf.compat.v1.train.AdamOptimizer, scope='orchestrate_agent'):

        # Agent 初始化方法的逐行注释

        # 初始化 Agent 实例
        Agent.__init__(self)

        # 设置 TensorFlow 会话
        self.sess = sess

        # 设置节点输入维度
        self.node_input_dim = node_input_dim

        # 设置集群输入维度
        self.cluster_input_dim = cluster_input_dim

        # 设置隐藏层维度列表
        self.hid_dims = hid_dims

        # 设置输出维度
        self.output_dim = output_dim

        # 设置最大深度
        self.max_depth = max_depth

        # 设置执行器层次
        self.executor_levels = executor_levels

        # 设置 epsilon（小的正数，用于数值稳定性）
        self.eps = eps

        # 设置激活函数
        self.act_fn = act_fn

        # 设置优化器
        self.optimizer = optimizer

        # 设置变量作用域
        self.scope = scope

        # 创建节点输入占位符
        self.node_inputs = tf.placeholder(tf.float32, [None, self.node_input_dim])

        # 创建集群输入占位符
        self.cluster_inputs = tf.placeholder(tf.float32, [None, self.cluster_input_dim])

        # 创建 GraphCNN 实例，处理节点输入
        self.gcn = GraphCNN(
            self.node_inputs, self.node_input_dim, self.hid_dims,
            self.output_dim, self.max_depth, self.act_fn, self.scope)

        # 创建 GraphSNN 实例，处理节点输入和 GraphCNN 输出的拼接
        self.gsn = GraphSNN(
            tf.concat([self.node_inputs, self.gcn.outputs], axis=1),
            self.node_input_dim + self.output_dim, self.hid_dims,
            self.output_dim, self.act_fn, self.scope)

        # Map gcn_outputs and raw_inputs to action probabilities
        # 将 GraphCNN 的输出和其他输入映射为节点和集群的动作概率
        self.node_act_probs, self.cluster_act_probs = self.orchestrate_network(
            self.node_inputs, self.gcn.outputs, self.cluster_inputs,
            self.gsn.summaries[0], self.gsn.summaries[1], self.act_fn)


        # Draw action based on the probability

        # 根据概率绘制动作
        # 总体来说，这段代码的作用是在给定节点动作的概率分布下，使用 Gumbel-Max 技巧从中采样确定实际执行的节点动作。
        # Gumbel-Max 技巧是一种通过引入 Gumbel 分布来从离散概率分布中采样的常见方法。

        # 对节点动作的概率取对数
        logits = tf.log(self.node_act_probs)
        # 生成均匀分布的随机噪声
        noise = tf.random_uniform(tf.shape(logits))
        # 通过 Gumbel-Max trick 抽取动作
        self.node_acts = tf.nn.top_k(logits - tf.log(-tf.log(noise)), k=3).indices

        # Cluster_acts
        logits = tf.log(self.cluster_act_probs)
        noise = tf.random_uniform(tf.shape(logits))
        self.cluster_acts = tf.nn.top_k(logits - tf.log(-tf.log(noise)), k=3).indices

        # 选择的动作
        # 这段代码的目的是计算在给定的节点和集群动作概率下，选择的动作对应的概率。
        # 这些概率计算对于计算策略梯度以及 Actor-Critic 方法中的更新步骤非常重要。
        # Advantage 和熵权重也被用于调整损失函数，以促使模型学到更好的策略。
        # 总体而言，这是强化学习中 Actor-Critic 类型算法中的一部分。

        # 创建节点动作向量的占位符
        self.node_act_vec = tf.placeholder(tf.float32, [None, None])

        # 创建集群动作向量的占位符
        self.cluster_act_vec = tf.placeholder(tf.float32, [None, None, None])

        # 创建 Advantage 的占位符
        # Advantage 通常用于衡量采取某个动作相对于平均动作的优势。
        self.adv = tf.placeholder(tf.float32, [None, 1])

        # 随时间衰减的熵权重
        self.entropy_weight = tf.placeholder(tf.float32, ())

        # 计算选择的节点动作概率
        self.selected_node_prob = tf.reduce_sum(tf.multiply(
            self.node_act_probs, self.node_act_vec),
            reduction_indices=1, keepdims=True)

        # 计算选择的集群动作概率
        self.selected_cluster_prob = tf.reduce_sum(tf.reduce_sum(tf.multiply(
            self.cluster_act_probs, self.cluster_act_vec),
            reduction_indices=2), reduction_indices=1, keep_dims=True)

        # Orchestrate loss due to advantge
        # 根据advantage计算由此引起的loss
        # 这样计算的目的是通过梯度下降来最小化损失，使模型更好地学习策略，
        # 使选择的动作的概率更趋近于 Advantage 的期望值。
        # 这是强化学习中策略梯度方法中常见的损失计算方式。
        self.adv_loss = tf.reduce_sum(tf.multiply(
            tf.log(self.selected_node_prob * self.selected_cluster_prob + \
                   self.eps), -self.adv))

        # Node_entropy 计算节点动作的熵
        # 其中 self.node_act_probs 是节点动作的概率分布。采用逐元素相乘的方式，
        # 然后对每个元素的对数概率进行求和。这个熵的计算有助于衡量节点动作分布的不确定性，
        # 可以在损失函数中用于控制策略的探索性质。添加的 self.eps 是为了数值稳定性。
        self.node_entropy = tf.reduce_sum(tf.multiply(
            self.node_act_probs, tf.log(self.node_act_probs + self.eps)))

        # Entropy loss
        # 这段代码用于计算熵损失，其中熵损失等于节点动作的熵。
        # 这个值可以用于增强模型的探索性，鼓励模型在学习过程中保持一定的不确定性。
        # 在这里，熵损失被赋值为 self.node_entropy，表示仅考虑节点动作的熵。
        # 注释中还提到了 self.cluster_entropy，但是在代码中被注释掉了，没有被使用。
        # 如果需要同时考虑集群动作的熵，可以取消注释并将其加到 self.entropy_loss 中。
        self.entropy_loss = self.node_entropy  # + self.cluster_entropy

        # Normalize entropy
        self.entropy_loss /= \
            (tf.log(tf.cast(tf.shape(self.node_act_probs)[1], tf.float32)) + \
             tf.log(float(len(self.executor_levels))))

        # Define combined loss  将熵损失标准化
        # 这样做的目的是使熵损失在不同模型和问题上的可比性更强，同时考虑了动作的数量和执行器层次的数量。
        self.act_loss = self.adv_loss + self.entropy_weight * self.entropy_loss

        # Get training parameters 获取训练参数
        # 使用 TensorFlow 函数获取模型中可训练的变量集合
        # 在训练过程中，优化器将根据损失函数对这些参数进行梯度下降更新。
        self.params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)

        # Operations for setting network parameters
        # 设置网络参数的操作
        self.input_params, self.set_params_op = \
            self.define_params_op()

        # Orchestrate gradients 用于计算动作损失相对于模型参数的梯度
        # 这段代码用于计算动作损失相对于模型参数的梯度。
        # 具体来说，使用 TensorFlow 的 tf.gradients 方法，传递动作损失 (self.act_loss) 和模型参数 (self.params)，
        # 以获取对于每个参数的梯度。这些梯度将用于优化器的更新步骤，通过梯度下降来最小化动作损失。
        # 这是强化学习中策略梯度方法的典型操作，用于调整模型参数以提高所选择的动作的概率。
        self.act_gradients = tf.gradients(self.act_loss, self.params)

        # Adaptive learning rate 用于接收自适应学习率的输入
        # 在模型训练时，通常会使用一种自适应学习率的策略，这样可以根据训练的进展动态地调整学习率的大小
        self.lr_rate = tf.placeholder(tf.float32, shape=[])

        # 编排优化器
        # 这段代码涉及到模型的优化器和参数更新的相关操作

        # 使用指定的优化器和学习率最小化动作损失
        self.act_opt = self.optimizer(self.lr_rate).minimize(self.act_loss)

        # 直接应用梯度来更新参数
        self.apply_grads = self.optimizer(self.lr_rate). \
            apply_gradients(zip(self.act_gradients, self.params))

        # 创建一个 TensorFlow Saver 对象，用于保存和加载模型参数
        self.saver = tf.compat.v1.train.Saver(max_to_keep=1000)

        # 运行 TensorFlow 图中的全局变量初始化操作
        self.sess.run(tf.global_variables_initializer())

    def orchestrate_network(self, node_inputs, gcn_outputs, cluster_inputs,
                            gsn_dag_summary, gsn_global_summary, act_fn):


        batch_size = 1
        node_inputs_reshape = tf.reshape(
            # 调整节点输入的形状
            # 这将使每个样本的节点输入被表示为一个矩阵，其中每行表示一个节点，每列表示节点特征。
            node_inputs, [batch_size, -1, self.node_input_dim])
        cluster_inputs_reshape = tf.reshape(  # 调整集群输入的形状
            cluster_inputs, [batch_size, -1, self.cluster_input_dim])
        gcn_outputs_reshape = tf.reshape(  # 调整 GCN 输出的形状
            gcn_outputs, [batch_size, -1, self.output_dim])
        gsn_dag_summ_reshape = tf.reshape(
            gsn_dag_summary, [batch_size, -1, self.output_dim])
        gsn_global_summ_reshape = tf.reshape(
            gsn_global_summary, [batch_size, -1, self.output_dim])
        gsn_global_summ_extend_cluster = tf.tile(  # 在集群维度上扩展 GSN 全局摘要
            gsn_global_summ_reshape, [1, tf.shape(gsn_dag_summ_reshape)[1], 1])

        with tf.variable_scope(self.scope):
            merge_node = tf.concat([
                node_inputs_reshape, gcn_outputs_reshape
            ], axis=2)  # 在指定的轴（这里是轴2）上连接张量。

            # 第一个隐藏层
            node_hid_0 = tl.fully_connected(merge_node, 32, activation_fn=act_fn)
            # 使用 TensoLayer（假设 tl.fully_connected 是 TensoLayer 的接口），定义一个全连接层。
            # 输入是 merge_node，包含32个神经元，激活函数为 act_fn。

            # 第二个隐藏层
            node_hid_1 = tl.fully_connected(node_hid_0, 16, activation_fn=act_fn)
            # 定义第二个全连接层，输入是第一个隐藏层的输出 node_hid_0，包含16个神经元，激活函数为 act_fn。

            # 第三个隐藏层
            node_hid_2 = tl.fully_connected(node_hid_1, 8, activation_fn=act_fn)
            # 定义第三个全连接层，输入是第二个隐藏层的输出 node_hid_1，包含8个神经元，激活函数为 act_fn。

            # 输出层
            node_outputs = tl.fully_connected(node_hid_2, 1, activation_fn=None)
            # 定义输出层，输入是第三个隐藏层的输出 node_hid_2，包含1个神经元，没有使用激活函数（线性输出）。
            # 通常用于回归问题，输出的值可以是任意实数。

            # Reshape the output dimension
            node_outputs = tf.reshape(node_outputs, [batch_size, -1])

            # Do softmax
            # 用于将网络输出转换为概率分布，以便更好地表示每个类别的概率。
            node_outputs = tf.nn.softmax(node_outputs, axis=-1)
            # 合并集群输入
            merge_cluster = tf.concat([cluster_inputs_reshape, ], axis=2)
            # 扩展状态
            expanded_state = expand_act_on_state(
                merge_cluster, [l / 50.0 for l in self.executor_levels])
            # 集群的第一个隐藏层
            cluster_hid_0 = tl.fully_connected(expanded_state, 32, activation_fn=act_fn)

            # 集群的第二个隐藏层
            cluster_hid_1 = tl.fully_connected(cluster_hid_0, 16, activation_fn=act_fn)

            # 集群的第三个隐藏层
            cluster_hid_2 = tl.fully_connected(cluster_hid_1, 8, activation_fn=act_fn)

            # 集群的输出层
            cluster_outputs = tl.fully_connected(cluster_hid_2, 1, activation_fn=None)

            # 调整集群输出的形状
            cluster_outputs = tf.reshape(cluster_outputs, [batch_size, -1])
            cluster_outputs = tf.reshape(
                cluster_outputs, [batch_size, -1, len(self.executor_levels)])

            # Do softmax 将cluster_outputs转换为概率分布
            # 在机器学习中，经过 softmax 操作的输出通常表示每个类别的概率，
            # 模型输出的最终结果可以是节点和集群的概率分布。
            cluster_outputs = tf.nn.softmax(cluster_outputs, dim=-1)
            return node_outputs, cluster_outputs

    # 这个操作通常用于应用梯度更新模型的参数。
    def apply_gradients(self, gradients, lr_rate):
        self.sess.run(self.apply_grads, feed_dict={
            i: d for i, d in zip(self.act_gradients + [self.lr_rate], gradients + [lr_rate])})

    def define_params_op(self):
        # 这个方法的目的是为模型的每个参数创建一个占位符，
        # 以便在后续的训练过程中可以向模型中注入新的参数值。
        # Define operations
        input_params = []
        # 为每个模型参数创建一个占位符
        for param in self.params:
            input_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        set_params_op = []
        # 针对每个参数和对应的占位符，创建一个操作，将占位符的值赋给相应的参数
        for idx, param in enumerate(input_params):
            set_params_op.append(self.params[idx].assign(param))
        return input_params, set_params_op

    def gcn_forward(self, node_inputs, summ_mats):
        # 这个方法的目的是在图神经网络（GraphSNN）中执行前向传播操作
        return self.sess.run([self.gsn.summaries],
                             feed_dict={i: d for i, d in
                                        zip([self.node_inputs] + self.gsn.summ_mats, [node_inputs] + summ_mats)})

    def get_params(self):
        # 获取模型的当前参数值
        # 用户可以通过调用这个方法来获取模型当前的权重和偏置等参数值。
        return self.sess.run(self.params)

    def save_model(self, file_path):
        # 模型参数被保存到了指定的文件路径，这可以在后续的时间点用于恢复模型或在其他地方使用。
        self.saver.save(self.sess, file_path)

    def update_gradients(self, node_inputs, cluster_inputs, node_act_vec, cluster_act_vec, adv, entropy_weight):
        # 从输入列表中获取单一的元素，以确保输入的是单一样本的信息
        node_inputs = node_inputs[0]
        cluster_inputs = cluster_inputs[0]
        node_act_vec = node_act_vec[0]
        cluster_act_vec = cluster_act_vec[0]
        entropy_weight = entropy_weight
        # 使用 TensorFlow 会话运行优化器操作，更新梯度
        self.sess.run(self.act_opt, feed_dict={i: d for i, d in zip(
            [self.node_inputs] + [self.cluster_inputs] + [self.node_act_vec] + [
                self.cluster_act_vec] + [self.adv] + [self.entropy_weight] + [self.lr_rate],
            [node_inputs] + [cluster_inputs] + [node_act_vec] + [cluster_act_vec] + [
                adv] + [entropy_weight] + [0.001])})

        # 使用 TensorFlow 会话运行损失计算操作，获取更新后的损失值
        loss_ = self.sess.run(self.act_loss, feed_dict={i: d for i, d in zip(
            [self.node_inputs] + [self.cluster_inputs] + [self.node_act_vec] + [
                self.cluster_act_vec] + [self.adv] + [self.entropy_weight],
            [node_inputs] + [cluster_inputs] + [node_act_vec] + [cluster_act_vec] + [
                adv] + [entropy_weight])})
        return loss_

    def predict(self, node_inputs, cluster_inputs):
        # 通过运行 TensorFlow 会话中的预测操作，获取模型对输入数据的预测结果。
        # 返回模型对输入数据的预测结果，
        # 包括节点激活概率、集群激活概率、节点动作和集群动作。
        return self.sess.run([self.node_act_probs, self.cluster_act_probs, self.node_acts, self.cluster_acts],
                             feed_dict={i: d for i, d in zip([self.node_inputs] + [self.cluster_inputs],
                                                             [node_inputs] + [cluster_inputs])})

    # 这个方法允许用户通过调用 set_params 方法，
    # 将新的参数值设置到模型中，以便在不重新训练的情况下更新模型的参数。
    def set_params(self, input_params):
        self.sess.run(self.set_params_op, feed_dict={
            i: d for i, d in zip(self.input_params, input_params)
        })

    def translate_state(self, obs):
        # 将观察到的状态信息转换为 NumPy 数组，传递给模型
        done_tasks, undone_tasks, curr_tasks_in_queue, deploy_state = obs
        done_tasks = np.array(done_tasks)
        undone_tasks = np.array(undone_tasks)
        curr_tasks_in_queue = np.array(curr_tasks_in_queue)
        deploy_state = np.array(deploy_state)

        # Compute total number of nodes 计算当前任务队列中节点的总数
        # 使用 Python 的 len 函数可以获取列表 curr_tasks_in_queue 的长度，即节点的总数。
        total_num_nodes = len(curr_tasks_in_queue)

        # Inputs to feed 准备要输入到模型中的数据
        # node_inputs 和 cluster_inputs 被初始化为零矩阵，用于存储节点和集群的输入数据
        # 这些零矩阵将在后续的代码中被填充为实际的输入数据，然后传递给模型进行预测或训练。
        node_inputs = np.zeros([total_num_nodes, self.node_input_dim])
        cluster_inputs = np.zeros([1, self.cluster_input_dim])

        for i in range(len(node_inputs)):
            # 根据给定的状态信息填充先前准备的节点和集群的输入数据

            # 对于每个节点，将当前任务队列的前12个元素和部署状态的前12个元素填充到节点输入
            node_inputs[i, :12] = curr_tasks_in_queue[i, :12]
            node_inputs[i, 12:] = deploy_state[i, :12]
            # 将已完成任务的前12个元素和未完成任务的前12个元素填充到集群输入 (cluster_inputs) 中。
        cluster_inputs[0, :12] = done_tasks[:12]
        cluster_inputs[0, 12:] = undone_tasks[:12]
        # 返回填充后的节点和集群的输入数据，这样这些数据可以用于模型的预测或训练。
        return node_inputs, cluster_inputs

    def get_valid_masks(self, cluster_states, frontier_nodes,
                        source_cluster, num_source_exec, exec_map, action_map):
        # 初始化集群和节点的有效掩码
        cluster_valid_mask = \
            np.zeros([1, len(cluster_states) * len(self.executor_levels)])
        cluster_valid = {}
        base = 0
        # 遍历集群状态，生成有效掩码
        for cluster_state in cluster_states:
            if cluster_state is source_cluster:
                least_exec_amount = \
                    exec_map[cluster_state] - num_source_exec + 1
            else:
                least_exec_amount = exec_map[cluster_state] + 1
            assert least_exec_amount > 0
            assert least_exec_amount <= self.executor_levels[-1] + 1
            # Find the index  寻找执行层级的索引
            exec_level_idx = bisect.bisect_left(
                self.executor_levels, least_exec_amount)
            # 更新集群有效性字典
            if exec_level_idx >= len(self.executor_levels):
                cluster_valid[cluster_state] = False
            else:
                cluster_valid[cluster_state] = True
            # 更新集群有效掩码
            for l in range(exec_level_idx, len(self.executor_levels)):
                cluster_valid_mask[0, base + l] = 1
            base += self.executor_levels[-1]
        # 计算总节点数
        total_num_nodes = int(np.sum(
            cluster_state.num_nodes for cluster_state in cluster_states))
        # 初始化节点有效掩码
        node_valid_mask = np.zeros([1, total_num_nodes])
        # 遍历前沿节点，生成节点有效掩码
        for node in frontier_nodes:
            if cluster_valid[node.cluster_state]:
                act = action_map.inverse_map[node]
                node_valid_mask[0, act] = 1
        # 返回生成的节点和集群有效掩码
        return node_valid_mask, cluster_valid_mask

    def invoke_model(self, obs):
        # Invoke learning model 调用学习模型
        node_inputs, cluster_inputs = self.translate_state(obs)  # obs观察状态
        # 使用模型进行预测
        node_act_probs, cluster_act_probs, node_acts, cluster_acts = \
            self.predict(node_inputs, cluster_inputs)
        # 返回模型的预测结果和输入数据
        return node_acts, cluster_acts, \
               node_act_probs, cluster_act_probs, \
               node_inputs, cluster_inputs

    def get_action(self, obs):
        # Parse observation 解析观察状态
        cluster_states, source_cluster, num_source_exec, \
        frontier_nodes, executor_limits, \
        exec_commit, moving_executors, action_map = obs
        # 如果前沿节点为空，则返回空动作和源执行器数量
        if len(frontier_nodes) == 0:
            return None, num_source_exec

        # Invoking the learning model 调用学习模型获取动作和相关信息
        node_act, cluster_act, \
        node_act_probs, cluster_act_probs, \
        node_inputs, cluster_inputs, \
        node_valid_mask, cluster_valid_mask, \
        gcn_mats, gcn_masks, summ_mats, \
        running_states_mat, state_summ_backward_map, \
        exec_map, cluster_states_changed = self.invoke_model(obs)

        # 如果所有节点的有效掩码都为零，则返回空动作和源执行器数量
        if sum(node_valid_mask[0, :]) == 0:
            return None, num_source_exec

        # Should be valid 应该是有效的
        assert node_valid_mask[0, node_act[0]] == 1

        # Parse node action 解析节点动作
        node = action_map[node_act[0]]
        cluster_idx = cluster_states.index(node.cluster_state)

        # Should be valid
        assert cluster_valid_mask[0, cluster_act[0, cluster_idx] +
                                  len(self.executor_levels) * cluster_idx] == 1
        # 计算代理执行动作
        if node.cluster_state is source_cluster:
            agent_exec_act = self.executor_levels[
                                 cluster_act[0, cluster_idx]] - \
                             exec_map[node.cluster_state] + \
                             num_source_exec
        else:
            agent_exec_act = self.executor_levels[
                                 cluster_act[0, cluster_idx]] - exec_map[node.cluster_state]

        # Parse  action 解析动作，计算使用的执行器数量
        use_exec = min(
            node.num_tasks - node.next_task_idx -
            exec_commit.node_commit[node] -
            moving_executors.count(node),
            agent_exec_act, num_source_exec)
        # 返回解析后的动作和使用的执行器数量
        return node, use_exec

    # 这个方法的目的是根据观察状态使用学习模型获取动作，
    # 并解析动作以确定节点和使用的执行器数量。
    # 首先，检查前沿节点是否为空，如果是，则返回空动作和源执行器数量。
    # 然后，调用invoke_model方法获取动作和相关信息。
    # 如果所有节点的有效掩码都为零，则返回空动作和源执行器数量。
    # 接着，解析节点动作和集群动作，并计算代理执行动作。
    # 最后，解析动作，计算使用的执行器数量，并返回结果。
