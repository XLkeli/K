import numpy as np
import tensorflow as tf
import random, os
from copy import deepcopy


# def fc(x, scope, nh, act=tf.nn.relu, init_scale=1.0):
#     # 这是一个用于创建全连接层的函数。通过指定输入 x、作用域 scope、隐藏单元数 nh 等参数，它会返回一个具有激活函数的全连接层。
#     # 这个函数的主要目的是创建神经网络中的全连接层，并通过激活函数对输出进行非线性变换。
#     # 这种模块化的设计使得在构建深度神经网络时能够方便地重复使用这样的层。
#
#     with tf.variable_scope(scope):
#         nin = x.get_shape()[1].value
#         w = tf.get_variable("w", [nin, nh],
#                             initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=0))
#         b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(0.0))
#         z = tf.matmul(x, w) + b
#         h = act(z)
#         return h

def fc(x, scope, nh, act=tf.nn.relu, init_scale=1.0):
    # 创建全连接层的函数，输入参数包括 x（输入张量）、scope（作用域名称）、nh（隐藏单元数）、act（激活函数，默认为 ReLU）、init_scale（权重初始化尺度，默认为 1.0）。

    with tf.variable_scope(scope):
        # 在给定的作用域内创建变量。

        nin = x.get_shape()[1].value
        # 获取输入张量 x 的第二维度，即输入的特征数。

        w = tf.compat.v1.get_variable("w", [nin, nh],
                            initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=0))
        # 创建权重变量 w，形状为 [nin, nh]，使用 Xavier 初始化器进行权重初始化。

        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(0.0))
        # 创建偏置变量 b，形状为 [nh]，并初始化为常数 0.0。

        z = tf.matmul(x, w) + b
        # 计算全连接层的线性变换，即 x * w + b。

        h = act(z)
        # 应用激活函数 act 对输出进行非线性变换，如果未指定激活函数，默认为 ReLU。

        return h


class Estimator:
    # 这个类定义了一个强化学习的估算器。它包含了价值函数和策略网络的构建，
    # 以及预测、采取动作等方法。在初始化时，建立了价值函数和策略网络的计算图。
    def __init__(self,
                 sess,
                 action_dim,
                 state_dim,
                 n_valid_node,
                 scope="estimator",
                 summaries_dir=None):
        self.sess = sess   # 保存传入的 TensorFlow 会话
        self.n_valid_node = n_valid_node   # 保存有效节点数目。
        self.action_dim = action_dim     # 保存动作的维度
        self.state_dim = state_dim       # 保存状态的维度
        self.scope = scope   # 保存变量作用域的名称，默认为 "estimator"。
        self.T = 144    # 保存时间步数。

        # Writes Tensorboard summaries to disk
        # 看似是用于构建和定义一个深度学习模型的损失函数
        self.summary_writer = None

        # 使用TensorFlow的变量作用域，将下面的操作放在指定的变量作用域中
        with tf.compat.v1.variable_scope(scope):
            value_loss = self._build_value_model()  # 构建价值模型，并将其损失值保存到value_loss

            # 在变量作用域"policy"下，构建策略模型，并将actor的损失和熵保存到actor_loss和entropy
            with tf.variable_scope("policy"):
                actor_loss, entropy = self._build_policy()

            # 计算最终的损失，包括actor的损失、值模型的损失、以及熵的负值
            self.loss = actor_loss + .5 * value_loss - 10 * entropy

        # Summaries
        self.summaries = tf.compat.v1.summary.merge([
            # 记录值损失（value_loss）的变化
            tf.compat.v1.summary.scalar("value_loss", self.value_loss),
            # 记录值输出（value_output）的均值变化
            tf.summary.scalar("value_output", tf.reduce_mean(self.value_output)),
        ])

        self.policy_summaries = tf.summary.merge([
            # 记录策略损失（policy_loss）的变化
            tf.summary.scalar("policy_loss", self.policy_loss),
            # 记录优势值（adv）的均值变化
            tf.summary.scalar("adv", tf.reduce_mean(self.tfadv)),
            # 记录熵（entropy）的变化
            tf.summary.scalar("entropy", self.entropy),
        ])

        # 如果给定了 summaries_dir，表示要写 Tensorboard 摘要
        if summaries_dir:
            # 根据给定的变量作用域创建一个摘要目录
            summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
            # 如果目录不存在，则创建
            if not os.path.exists(summary_dir):
                os.makedirs(summary_dir)
            # 创建一个 Tensorboard 摘要写入器
            self.summary_writer = tf.compat.v1.summary.FileWriter(summary_dir)

        # 初始化邻居列表
        self.neighbors_list = [[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6]]

    def _build_value_model(self):    # 定义构建价值模型的方法

        # 创建占位符，表示输入的状态
        # shape=[None, self.state_dim] 其中None表示可以处理任意数量的样本
        self.state = X = tf.compat.v1.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name="X")

        # 创建占位符，表示 Temporal Difference (TD) 的目标值
        # The TD target value
        self.y_pl = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="y")
        # 创建占位符，表示学习率
        self.loss_lr = tf.placeholder(tf.float32, None, "learning_rate")

        # 构建一个具有3个隐藏层的神经网络
        # 3 layers network.
        l1 = fc(X, "l1", 128, act=tf.nn.relu)  # 构建具有 128 个神经元的第一个全连接层，并使用ReLU激活函数
        l2 = fc(l1, "l2", 64, act=tf.nn.relu)  # 构建具有 64 个神经元的第二个全连接层，并使用 ReLU 激活函数。
        l3 = fc(l2, "l3", 32, act=tf.nn.relu)  # 构建具有 32 个神经元的第一个全连接层，并使用 ReLU 激活函数。

        # 构建值模型的输出层，输出为一个值，使用 ReLU 激活函数
        self.value_output = fc(l3, "value_output", 1, act=tf.nn.relu)
        # 定义值模型的损失函数，使用均方误差（MSE）来衡量模型输出与目标值之间的差异
        self.value_loss = tf.reduce_mean(tf.squared_difference(self.y_pl, self.value_output))
        # 定义值模型的训练操作，使用Adam优化器来最小化损失
        self.value_train_op = tf.train.AdamOptimizer(self.loss_lr).minimize(self.value_loss)

        # 返回值模型的损失
        return self.value_loss

    # 定义构建策略模型的方法
    # 包含了输入状态、动作、优势值、邻居蒙版等占位符，
    # 以及一个有三个隐藏层的神经网络。然后计算了 logits，根据邻居蒙版生成了有效的 logits。
    # 接着计算了 softmaxprob 和 logsoftmaxprob
    # 最后计算了 actor 损失、策略的熵和最终的策略损失。
    # 策略模型的训练操作使用了 Adam 优化器。
    # 最后返回了 actor 损失和熵这两个值。
    def _build_policy(self):

        # 创建占位符，表示策略模型的输入状态
        self.policy_state = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name="P")

        # 创建占位符，表示动作
        self.ACTION = tf.placeholder(shape=[None, self.action_dim], dtype=tf.float32, name="action")

        # 创建占位符，表示优势值
        self.tfadv = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='advantage')

        # 创建占位符，表示邻居蒙版
        self.neighbor_mask = tf.placeholder(shape=[None, self.action_dim], dtype=tf.float32, name="neighbormask")

        # 构建神经网络，具有3个隐藏层
        l1 = fc(self.policy_state, "l1", 128, act=tf.nn.relu)
        l2 = fc(l1, "l2", 64, act=tf.nn.relu)
        l3 = fc(l2, "l3", 32, act=tf.nn.relu)

        # 避免 valid_logits 全为零
        self.logits = logits = fc(l3, "logits", self.action_dim, act=tf.nn.relu) + 1

        # 根据邻居蒙版生成有效的 logits
        self.valid_logits = logits * self.neighbor_mask

        # 计算 softmaxprob 和 logsoftmaxprob
        self.softmaxprob = tf.nn.softmax(tf.log(self.valid_logits + 1e-8))
        self.logsoftmaxprob = tf.nn.log_softmax(self.softmaxprob)

        # 计算负对数概率，用于计算 actor 损失
        self.neglogprob = - self.logsoftmaxprob * self.ACTION

        # 计算 actor 损失
        self.actor_loss = tf.reduce_mean(tf.reduce_sum(self.neglogprob * self.tfadv, axis=1))

        # 计算策略的熵
        self.entropy = - tf.reduce_mean(self.softmaxprob * self.logsoftmaxprob)

        # 计算最终的策略损失，包括 actor 损失和熵的调和平均
        self.policy_loss = self.actor_loss - 0.01 * self.entropy

        # 定义策略模型的训练操作，使用 Adam 优化器
        self.policy_train_op = tf.train.AdamOptimizer(self.loss_lr).minimize(self.policy_loss)

        # 返回 actor 损失和熵
        return self.actor_loss, self.entropy

    # 定义了一个预测方法，用于获取给定状态 s 的值模型输出。
    def predict(self, s):
        # 使用 TensorFlow 的会话运行值模型，得到状态 s 对应的值模型输出
        # 这一行代码实际上是在进行前向传播，得到模型在输入状态 s 下的预测输出。
        value_output = self.sess.run(self.value_output, {self.state: s})

        # 返回值模型的输出
        # 将值模型的输出作为预测的结果返回。这个输出可能表示状态 s 的估计值，
        # 根据模型的设计，可能是关于状态的某种值的估计。
        return value_output

    def action(self, s, ava_node, context, epsilon):
        # 该方法用于选择动作，同时为策略梯度的训练收集相关的数据。
        # 在方法中，首先运行值模型获取状态对应的值输出，
        # 然后根据策略模型采样动作。
        # 最后，将选择的动作和相关信息返回。

        # 运行值模型，获取状态 s 对应的值模型输出
        value_output = self.sess.run(self.value_output, {self.state: s}).flatten()
        action_tuple = []  # 存储最终选择的动作
        valid_prob = []  # 存储有效动作的概率

        # 用于训练策略梯度的相关数据
        action_choosen_mat = []  # 存储选择的动作的矩阵表示
        policy_state = []  # 存储状态，用于策略梯度的训练
        curr_state_value = []  # 存储当前状态对应的值模型输出
        next_state_ids = []  # 存储下一个状态的ID

        grid_ids = [x for x in range(self.n_valid_node)]

        # 初始化有效动作的蒙版
        self.valid_action_mask = np.zeros((self.n_valid_node, self.action_dim))
        for i in range(len(ava_node)):
            for j in ava_node[i]:
                self.valid_action_mask[i][j] = 1
        curr_neighbor_mask = deepcopy(self.valid_action_mask)

        self.valid_neighbor_node_id = [[i for i in range(self.action_dim)], [i for i in range(self.action_dim)]]

        # 计算策略概率
        action_probs = self.sess.run(self.softmaxprob, {self.policy_state: s,
                                                        self.neighbor_mask: curr_neighbor_mask})
        curr_neighbor_mask_policy = []

        # 采样动作
        for idx, grid_valid_idx in enumerate(grid_ids):
            action_prob = action_probs[idx]

            # 记录动作概率
            valid_prob.append(action_prob)

            # 如果上下文为0，直接跳过选择动作的部分
            if int(context[idx]) == 0:
                continue

            # 使用采样方法选择动作
            curr_action_indices_temp = np.random.choice(self.action_dim, int(context[idx]),
                                                        p=action_prob / np.sum(action_prob))
            curr_action_indices = [0] * self.action_dim
            for kk in curr_action_indices_temp:
                curr_action_indices[kk] += 1

            # 记录选择的动作
            self.valid_neighbor_grid_id = self.valid_neighbor_node_id
            for curr_action_idx, num_driver in enumerate(curr_action_indices):
                if num_driver > 0:
                    # 获取选择动作对应的节点 ID
                    end_node_id = int(self.valid_neighbor_node_id[grid_valid_idx][curr_action_idx])
                    action_tuple.append(end_node_id)

                    # 记录训练所需的相关信息
                    temp_a = np.zeros(self.action_dim)
                    temp_a[curr_action_idx] = 1
                    action_choosen_mat.append(temp_a)
                    policy_state.append(s[idx])
                    curr_state_value.append(value_output[idx])
                    next_state_ids.append(self.valid_neighbor_grid_id[grid_valid_idx][curr_action_idx])
                    curr_neighbor_mask_policy.append(curr_neighbor_mask[idx])

        return action_tuple, np.stack(valid_prob), \
            np.stack(policy_state), np.stack(action_choosen_mat), curr_state_value, \
            np.stack(curr_neighbor_mask_policy), next_state_ids

    def compute_advantage(self, curr_state_value, next_state_ids, next_state, node_reward, gamma):
        # compute advantage
        advantage = []
        node_reward = node_reward.flatten()   # 将 node_reward 展平成一维数组

        # 获取下一个状态 next_state 的值模型输出，将其展平成一维数组。
        qvalue_next = self.sess.run(self.value_output, {self.state: next_state}).flatten()   # 获取下一个状态的值模型输出
        for idx, next_state_id in enumerate(next_state_ids):  # 对每个下一个状态进行循环遍历
            # 计算每个状态的优势
            temp_adv = sum(node_reward) + gamma * sum(qvalue_next) - curr_state_value[idx]
            advantage.append(temp_adv)  # 将计算得到的优势添加到 advantage 列表中
        return advantage

    def compute_targets(self, valid_prob, next_state, node_reward, gamma):
        # 计算目标值
        targets = []
        node_reward = node_reward.flatten()  # 将 node_reward 展平成一维数组
        qvalue_next = self.sess.run(self.value_output, {self.state: next_state}).flatten()  # 获取下一个状态的值模型输出

        for idx in np.arange(self.n_valid_node):
            # 获取有效动作的概率
            grid_prob = valid_prob[idx][self.valid_action_mask[idx] > 0]

            # 计算当前网格的目标值
            curr_grid_target = np.sum(grid_prob * (sum(node_reward) + gamma * sum(qvalue_next)))
            targets.append(curr_grid_target)

        # 将计算得到的目标值转换成数组并重塑形状
        return np.array(targets).reshape([-1, 1])

    # 定义了一个初始化方法，用于训练值模型。
    def initialization(self, s, y, learning_rate):
        # 获取当前模型的 TensorFlow 会话
        sess = self.sess

        # 构建用于喂入数据的字典，包含输入状态 s、目标值 y 和学习率 learning_rate。
        feed_dict = {self.state: s, self.y_pl: y, self.loss_lr: learning_rate}

        # 运行值模型的训练操作和损失计算
        _, value_loss = sess.run([self.value_train_op, self.value_loss], feed_dict)

        # 返回值模型的损失，这个损失值在训练过程中可能被用于监控和调整模型
        return value_loss

    #  定义了一个更新策略模型的方法。
    def update_policy(self, policy_state, advantage, action_choosen_mat, curr_neighbor_mask, learning_rate,
                      global_step):
        sess = self.sess  # 获取当前模型的 TensorFlow 会话。

        # 构建一个用于喂入数据的字典，包含策略模型的输入数据，优势值，选择动作的矩阵表示，当前邻居蒙版，以及学习率。
        feed_dict = {self.policy_state: policy_state,
                     self.tfadv: advantage,
                     self.ACTION: action_choosen_mat,
                     self.neighbor_mask: curr_neighbor_mask,
                     self.loss_lr: learning_rate}
        summaries, _, loss = sess.run([self.policy_summaries, self.policy_train_op, self.policy_loss], feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
            self.summary_writer.flush()
        # 返回策略模型的损失，这个损失值在训练过程中可能被用于监控和调整模型。

       # print(loss)
        return loss



    def update_value(self, s, y, learning_rate, global_step):
        sess = self.sess
        feed_dict = {self.state: s, self.y_pl: y, self.loss_lr: learning_rate}
        summaries, _, loss = sess.run([self.summaries, self.value_train_op, self.value_loss], feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
            self.summary_writer.flush()
        return loss


class policyReplayMemory:
    def __init__(self, memory_size, batch_size):
        # 初始化策略回放内存
        self.states = []  # 初始化一个空列表，用于存储状态
        self.neighbor_mask = []  # 存储邻居蒙版
        self.actions = []  # 存储动作
        self.rewards = []  # 存储奖励
        self.batch_size = batch_size  # 批处理大小
        self.memory_size = memory_size  # 回放内存大小
        self.current = 0  # 当前索引
        self.curr_lens = 0  # 当前长度

    # Put data in policy replay memory
    def add(self, s, a, r, mask):
        # 添加经验样本到策略回放内存

        if self.curr_lens == 0:
            # 如果当前长度为0，直接赋值
            self.states = s
            self.actions = a
            self.rewards = r
            self.neighbor_mask = mask
            self.curr_lens = self.states.shape[0]

        elif self.curr_lens <= self.memory_size:
            # 如果当前长度小于等于回放内存大小，拼接到末尾
            self.states = np.concatenate((self.states, s), axis=0)
            self.neighbor_mask = np.concatenate((self.neighbor_mask, mask), axis=0)
            self.actions = np.concatenate((self.actions, a), axis=0)
            self.rewards = np.concatenate((self.rewards, r), axis=0)
            self.curr_lens = self.states.shape[0]
        else:
            # 如果当前长度大于回放内存大小，替换随机位置的样本
            new_sample_lens = s.shape[0]
            index = random.randint(0, self.curr_lens - new_sample_lens)
            self.states[index:(index + new_sample_lens)] = s
            self.actions[index:(index + new_sample_lens)] = a
            self.rewards[index:(index + new_sample_lens)] = r
            self.neighbor_mask[index:(index + new_sample_lens)] = mask

    # Take a batch of samples
    def sample(self):
        # 从策略回放内存中抽样

        if self.curr_lens <= self.batch_size:
            # 如果当前长度小于等于批处理大小，返回全部样本
            return [self.states, self.actions, np.array(self.rewards), self.neighbor_mask]
        else:
            # 否则，随机抽取批处理大小的样本
            indices = random.sample(list(range(0, self.curr_lens)), self.batch_size)
            batch_s = self.states[indices]
            batch_a = self.actions[indices]
            batch_r = self.rewards[indices]
            batch_mask = self.neighbor_mask[indices]
            return [batch_s, batch_a, batch_r, batch_mask]

    def reset(self):
        # 重置策略回放内存

        self.states = []  # 重置状态列表
        self.actions = []  # 重置动作列表
        self.rewards = []  # 重置奖励列表
        self.neighbor_mask = []  # 重置邻居蒙版列表
        self.curr_lens = 0  # 重置当前长度为0


class ReplayMemory:
    def __init__(self, memory_size, batch_size):
        # 初始化回放内存
        self.states = []  # 存储当前状态
        self.next_states = []  # 存储下一个状态
        self.actions = []  # 存储动作
        self.rewards = []  # 存储奖励

        self.batch_size = batch_size  # 批处理大小
        self.memory_size = memory_size  # 回放内存大小
        self.current = 0  # 当前索引
        self.curr_lens = 0  # 当前长度

    # Put data in policy replay memory
    def add(self, s, a, r, next_s):
        # 添加经验样本到回放内存
        if self.curr_lens == 0:
            # 如果当前长度为0，直接赋值
            self.states = s
            self.actions = a
            self.rewards = r
            self.next_states = next_s
            self.curr_lens = self.states.shape[0]

        elif self.curr_lens <= self.memory_size:
            # 如果当前长度小于等于回放内存大小，拼接到末尾
            self.states = np.concatenate((self.states, s), axis=0)
            self.next_states = np.concatenate((self.next_states, next_s), axis=0)
            self.actions = np.concatenate((self.actions, a), axis=0)
            self.rewards = np.concatenate((self.rewards, r), axis=0)
            self.curr_lens = self.states.shape[0]
        else:
            # 如果当前长度大于回放内存大小，替换随机位置的样本
            new_sample_lens = s.shape[0]
            index = random.randint(0, self.curr_lens - new_sample_lens)
            self.states[index:(index + new_sample_lens)] = s
            self.actions[index:(index + new_sample_lens)] = a
            self.rewards[index:(index + new_sample_lens)] = r
            self.next_states[index:(index + new_sample_lens)] = next_s

    # Take a batch of samples
    def sample(self):
        # 从回放内存中抽样

        if self.curr_lens <= self.batch_size:
            # 如果当前长度小于等于批处理大小，返回全部样本
            return [self.states, self.actions, self.rewards, self.next_states]
        else:
            # 否则，随机抽取批处理大小的样本
            indices = random.sample(list(range(0, self.curr_lens)), self.batch_size)
            batch_s = self.states[indices]
            batch_a = self.actions[indices]
            batch_r = self.rewards[indices]
            batch_mask = self.next_states[indices]
            return [batch_s, batch_a, batch_r, batch_mask]

    def reset(self):
        # 重置回放内存

        self.states = []  # 重置状态列表
        self.actions = []  # 重置动作列表
        self.rewards = []  # 重置奖励列表
        self.next_states = []  # 重置下一个状态列表
        self.curr_lens = 0  # 重置当前长度为0


class ModelParametersCopier():
    # 这个类用于将一个模型的参数复制到另一个模型中。主要是通过 TensorFlow 的变量来实现的。

    def __init__(self, estimator1, estimator2):
        # 构造函数，初始化模型参数复制器
        e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
        e1_params = sorted(e1_params, key=lambda v: v.name)
        e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
        e2_params = sorted(e2_params, key=lambda v: v.name)

        self.update_ops = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            # 遍历两个模型的参数，创建更新操作
            op = e2_v.assign(e1_v)
            self.update_ops.append(op)

    def make(self, sess):
        sess.run(self.update_ops)

    # 整个代码主要涉及强化学习模型的构建和训练，包括值函数、策略网络，
    # 以及用于经验回放的回放内存。
    # 这样的模型可以在强化学习任务中进行学习和决策