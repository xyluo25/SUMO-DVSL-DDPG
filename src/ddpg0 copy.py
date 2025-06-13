

from __future__ import division
from networks0 import rm_vsl_co
import traci
# from Priority_Replay import SumTree, Memory
import tensorflow as tf
import numpy as np
import time

import os
import sys

tools = '/usr/share/sumo/tools'
tools = r"C:/Program Files (x86)/Eclipse/Sumo/tools"
sys.path.append(tools)


# change the working dir to current dir
os.chdir(os.getcwd())

EP_MAX = 600
LR_A = 0.0002    # learning rate for actor
LR_C = 0.0005    # learning rate for critic
GAMMA = 0.9      # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 64
BATCH_SIZE = 32

RENDER = False

###############################  DDPG  ####################################


class Actor(tf.keras.Model):
    def __init__(self, s_dim, a_dim):
        super(Actor, self).__init__()
        # Layer sizes mirror the original TF1 implementation: 60 units, then output dimension a_dim
        self.fc1 = tf.keras.layers.Dense(60, activation='relu',
                                         kernel_initializer='glorot_uniform',
                                         name='actor_l1')
        # Output layer with sigmoid activation, no bias, to be scaled by 8
        self.fc2 = tf.keras.layers.Dense(a_dim, activation='sigmoid',
                                         use_bias=False,
                                         kernel_initializer='glorot_uniform',
                                         name='actor_l2')

    def call(self, s):
        x = self.fc1(s)
        a_raw = self.fc2(x)
        # Scale to match original range: sigmoid outputs in [0,1], so multiply by 8
        return a_raw * 8.0


class Critic(tf.keras.Model):
    def __init__(self, s_dim, a_dim):
        super(Critic, self).__init__()
        # First layer combines state and action with separate weight matrices and a bias
        n_l1 = 50
        # Equivalent to original w1_s and w1_a without biases
        self.w1_s = tf.keras.layers.Dense(n_l1, use_bias=False,
                                          kernel_initializer='glorot_uniform',
                                          name='critic_w1_s')
        self.w1_a = tf.keras.layers.Dense(n_l1, use_bias=False,
                                          kernel_initializer='glorot_uniform',
                                          name='critic_w1_a')
        # Bias vector b1 of shape [n_l1]
        self.b1 = tf.Variable(tf.zeros(shape=(n_l1,)),
                              trainable=True, name='critic_b1')
        # Final output layer to produce Q-value
        self.q_out = tf.keras.layers.Dense(1,
                                           kernel_initializer='glorot_uniform',
                                           name='critic_q')

    def call(self, s, a):
        # s: [batch, s_dim], a: [batch, a_dim]
        net_s = self.w1_s(s)         # [batch, n_l1]
        net_a = self.w1_a(a)         # [batch, n_l1]
        net = tf.nn.relu(net_s + net_a + self.b1)  # [batch, n_l1]
        q = self.q_out(net)          # [batch, 1]
        return q


class VSL_DDPG_PR:
    def __init__(self, a_dim, s_dim):
        self.a_dim = a_dim
        self.s_dim = s_dim

        # Experience replay memory: numpy array of shape (MEMORY_CAPACITY, s_dim*2 + a_dim + 1)
        self.memory = np.zeros(
            (MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0

        # Create the actor and critic networks
        self.actor = Actor(s_dim, a_dim)
        self.critic = Critic(s_dim, a_dim)

        # Create target networks
        self.actor_target = Actor(s_dim, a_dim)
        self.critic_target = Critic(s_dim, a_dim)

        # Initialize target networks with the same weights as the originals
        # This must be done once (after the models are built)
        dummy_state = tf.zeros((1, s_dim))
        dummy_action = tf.zeros((1, a_dim))
        _ = self.actor(dummy_state)
        _ = self.critic(dummy_state, dummy_action)
        _ = self.actor_target(dummy_state)
        _ = self.critic_target(dummy_state, dummy_action)

        self._update_target_weights(tau=1.0)  # hard update for initialization

        # Optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=LR_A)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=LR_C)

        # For saving and loading
        self.ckpt = tf.train.Checkpoint(actor=self.actor,
                                        critic=self.critic,
                                        actor_target=self.actor_target,
                                        critic_target=self.critic_target,
                                        actor_opt=self.actor_optimizer,
                                        critic_opt=self.critic_optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt,
                                                       directory='ddpg_checkpoints',
                                                       max_to_keep=1)

    def _update_target_weights(self, tau=None):
        """
        Soft-update or hard-update (if tau=1.0) target network weights:
        target_var = tau * main_var + (1 - tau) * target_var
        """
        if tau is None:
            tau = TAU
        # Actor weights
        actor_weights = self.actor.trainable_variables
        actor_target_weights = self.actor_target.trainable_variables
        for w, w_t in zip(actor_weights, actor_target_weights):
            w_t.assign(tau * w + (1.0 - tau) * w_t)
        # Critic weights
        critic_weights = self.critic.trainable_variables
        critic_target_weights = self.critic_target.trainable_variables
        for w, w_t in zip(critic_weights, critic_target_weights):
            w_t.assign(tau * w + (1.0 - tau) * w_t)

    def choose_action(self, s):
        """
        Given a single state s (shape: [s_dim, ]), return a single action (shape: [a_dim, ]).
        """
        s = tf.convert_to_tensor(
            s.reshape(1, -1), dtype=tf.float32)  # [1, s_dim]
        a = self.actor(s)  # [1, a_dim]
        return a.numpy()[0]

    def store_transition(self, s, a, r, s_):
        """
        Store one transition in replay memory.
        s:   [s_dim, ]
        a:   [a_dim, ]
        r:   scalar reward
        s_:  [s_dim, ]
        """
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1

    @tf.function
    def _train_step(self, bs, ba, br, bs_):
        """
        One training step for both actor and critic. Uses @tf.function for graph execution.
        bs:  [batch, s_dim]
        ba:  [batch, a_dim]
        br:  [batch, 1]
        bs_: [batch, s_dim]
        """
        # Compute target actions and target Q-values
        a_target = self.actor_target(bs_)                   # [batch, a_dim]
        q_target_next = self.critic_target(bs_, a_target)   # [batch, 1]
        q_target = br + GAMMA * q_target_next               # [batch, 1]

        # Critic update: minimize MSE between q(bs, ba) and q_target
        with tf.GradientTape() as tape_c:
            q_eval = self.critic(bs, ba)                    # [batch, 1]
            critic_loss = tf.math.reduce_mean(tf.square(q_target - q_eval))
        critic_grads = tape_c.gradient(
            critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grads, self.critic.trainable_variables))

        # Actor update: maximize expected Q-value from critic (i.e., minimize -Q)
        with tf.GradientTape() as tape_a:
            a_eval = self.actor(bs)                         # [batch, a_dim]
            q_for_a = self.critic(bs, a_eval)               # [batch, 1]
            # We want to maximize Q, so minimize -mean(Q)
            actor_loss = -tf.math.reduce_mean(q_for_a)
        actor_grads = tape_a.gradient(
            actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grads, self.actor.trainable_variables))

        # Soft-update target networks
        self._update_target_weights()

    def learn(self):
        """
        Sample a batch from memory and perform one training step.
        """
        # Randomly sample a batch of transitions
        indices = np.random.choice(
            min(self.pointer, MEMORY_CAPACITY), size=BATCH_SIZE)
        bt = self.memory[indices, :]

        bs = bt[:, :self.s_dim]                                # [batch, s_dim]
        ba = bt[:, self.s_dim:self.s_dim + self.a_dim]         # [batch, a_dim]
        br = bt[:, self.s_dim + self.a_dim:self.s_dim +
                self.a_dim + 1]  # [batch, 1]
        # [batch, s_dim]
        bs_ = bt[:, -self.s_dim:]

        # Convert to tensors
        bs = tf.convert_to_tensor(bs, dtype=tf.float32)
        ba = tf.convert_to_tensor(ba, dtype=tf.float32)
        br = tf.convert_to_tensor(br, dtype=tf.float32)
        bs_ = tf.convert_to_tensor(bs_, dtype=tf.float32)

        # Perform a combined train step
        self._train_step(bs, ba, br, bs_)

    def savemodel(self):
        """
        Save the entire DDPG (actor, critic, their target networks, and optimizers) to a checkpoint.
        """
        self.ckpt_manager.save()

    def loadmodel(self):
        """
        Restore from the latest checkpoint if available.
        """
        latest_ckpt = self.ckpt_manager.latest_checkpoint
        if latest_ckpt:
            self.ckpt.restore(latest_ckpt)
        else:
            print("No checkpoint found. Starting from scratch.")


class VSL_DDPG_PR_(object):
    def __init__(self, a_dim, s_dim,):
        # self.memory = Memory(capacity=MEMORY_CAPACITY)
        self.memory = np.zeros(
            (MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim = a_dim, s_dim
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.a = self._build_a(self.S,)
        q = self._build_c(self.S, self.a, )
        a_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(
            decay=1 - TAU)          # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(
            c_params)]      # soft update operation
        # replaced target parameters
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(
            LR_A).minimize(a_loss, var_list=a_params)
        self.td = self.R + GAMMA * q_ - q

        # soft replacement happened at here
        with tf.control_dependencies(target_update):
            q_target = self.R + GAMMA * q_
            td_error = tf.losses.mean_squared_error(
                labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(
                LR_C).minimize(td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=1)

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        #        tree_idx, bt, ISWeights = self.memory.sample(BATCH_SIZE)
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs,
                      self.a: ba, self.R: br, self.S_: bs_})


#    def store_transition(self, s, a, r, s_):
#        transition = np.hstack((s, a, r, s_))
#        self.memory.store(transition)


    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        # replace the old memory with new memory
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            neta = tf.layers.dense(
                s, 60, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(neta, self.a_dim, activation=tf.nn.sigmoid,
                                name='l2', trainable=trainable,  use_bias=False)
            return tf.multiply(a, 8, name='scaled_a')

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 50
            w1_s = tf.get_variable(
                'w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable(
                'w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            netc = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(netc, 1, trainable=trainable)

    def savemodel(self,):
        self.saver.save(
            self.sess, 'ddpg_networkss_withoutexplore/' + 'ddpg.ckpt')

    def loadmodel(self,):
        loader = tf.train.import_meta_graph(
            'ddpg_networkss_withoutexplore/ddpg.ckpt.meta')
        loader.restore(self.sess, tf.train.latest_checkpoint(
            "ddpg_networkss_withoutexplore/"))


def from_a_to_mlv(a):
    return 17.8816 + 2.2352*np.floor(a)


vsl_controller = VSL_DDPG_PR(s_dim=13, a_dim=5)
net = rm_vsl_co(visualization=True)
total_step = 0
var = 1.5
att = []
all_ep_r = []
att = []
all_co = []
all_hc = []
all_nox = []
all_pmx = []
all_oflow = []
all_bspeed = []
stime = np.zeros(13,)
co = 0
hc = 0
nox = 0
pmx = 0
oflow = 0
bspeed = 0
traveltime = 'meanTravelTime='
for ep in range(EP_MAX):
    time_start = time.time()
    co = 0
    hc = 0
    nox = 0
    pmx = 0
    ep_r = 0
    oflow = 0
    bspeed = 0
    v = 29.06*np.ones(5,)
    net.start_new_simulation(write_newtrips=False)
    s, r, simulationSteps, oflow_temp, bspeed_temp, co_temp, hc_temp, nox_temp, pmx_temp = net.run_step(
        v)
    co = co + co_temp
    hc = hc + hc_temp
    nox = nox + nox_temp
    pmx = pmx + pmx_temp
    oflow = oflow + oflow_temp
    bspeed_temp = bspeed + bspeed_temp
    stime[0:12] = s
    stime[12] = 0
    while simulationSteps < 18000:
        a = vsl_controller.choose_action(stime)
        # a = np.clip(np.random.laplace(a, var), 0, 7.99) The exploration is not very useful
        v = from_a_to_mlv(a)
        stime_ = np.zeros(13,)
        s_, r, simulationSteps, oflow_temp, bspeed_temp, co_temp, hc_temp, nox_temp, pmx_temp = net.run_step(
            v)

#        vid_list = traci.lane.getLastStepVehicleIDs('m7_5') + traci.lane.getLastStepVehicleIDs('m7_4') + traci.lane.getLastStepVehicleIDs('m6_4') + traci.lane.getLastStepVehicleIDs('m6_3')
#        for i in range(len(vid_list)):
#            traci.vehicle.setLaneChangeMode(vid_list[i], 0b001000000000)
        co = co + co_temp
        hc = hc + hc_temp
        nox = nox + nox_temp
        pmx = pmx + pmx_temp
        oflow = oflow + oflow_temp
        bspeed = bspeed + bspeed_temp
        stime_[0:12] = s_
        stime_[12] = simulationSteps/18000
        vsl_controller.store_transition(stime, a, r, stime_)
        total_step = total_step + 1
        if total_step > MEMORY_CAPACITY:
            # var = abs(1.5 - 1.5/600*ep)    # decay the action randomness
            vsl_controller.learn()
        stime = stime_
        ep_r += r
    all_ep_r.append(ep_r)
    all_co.append(co/1000)
    all_hc.append(hc/1000)
    all_nox.append(nox/1000)
    all_pmx.append(pmx/1000)
    all_oflow.append(oflow)
    all_bspeed.append(bspeed/300)
    net.close()
    fname = 'output_sumo.xml'
    with open(fname, 'r') as f:  # 打开文件
        lines = f.readlines()  # 读取所有行
        last_line = lines[-2]  # 取最后一行
    nPos = last_line.index(traveltime)
    aat_tempo = float(last_line[nPos+16:nPos+21])
    print('Episode:', ep, ' Rewards: %.4f' % ep_r, 'CO(g): %.4f' % co,
          'HC(g): %.4f' % hc, 'NOX(g): %.4f' % nox, 'PMX(g): %.4f' % pmx, 'Out-in flow: %.4f' % oflow,
          'Bottleneck speed: %.4f' % bspeed, 'Average travel time: %.4f' % aat_tempo)
    if all_ep_r[ep] == max(all_ep_r) and ep > 15:
        vsl_controller.savemodel()
    time_end = time.time()
    print('totally cost', time_end-time_start)


'''
Comparison with no VSL control
'''

# time_start=time.time()
# vsl_controller = VSL_DDPG_PR(s_dim = 13, a_dim = 5)
# net = rm_vsl_co(visualization = False, incidents = False)
# net.writenewtrips()
# traveltime='meanTravelTime='
# co = 0
# hc = 0
# nox = 0
# pmx = 0
# ep_r = 0
# oflow = 0
# bspeed = 0
# v = 29.06*np.ones(5,)
# net.start_new_simulation(write_newtrips = False)
# s, r, simulationSteps, oflow_temp, bspeed_temp, co_temp, hc_temp, nox_temp, pmx_temp = net.run_step(v)
# co = co + co_temp
# hc = hc + hc_temp
# nox = nox + nox_temp
# pmx = pmx + pmx_temp
# oflow = oflow + oflow_temp
# bspeed_temp = bspeed + bspeed_temp
# while simulationSteps < 18000:
#    s_, r, simulationSteps, oflow_temp, bspeed_temp, co_temp, hc_temp, nox_temp, pmx_temp = net.run_step(v)
#    co = co + co_temp
#    hc = hc + hc_temp
#    nox = nox + nox_temp
#    pmx = pmx + pmx_temp
#    oflow = oflow + oflow_temp
#    bspeed = bspeed + bspeed_temp
#    ep_r += r
# net.close()
# fname = 'output_sumo.xml'
# with open(fname, 'r') as f:  # 打开文件
#    lines = f.readlines()  # 读取所有行
#    last_line = lines[-2]  # 取最后一行
# nPos=last_line.index(traveltime)
# aat_tempo = float(last_line[nPos+16:nPos+21])
# print( 'Average Travel Time: %.4f' % aat_tempo, ' Rewards: %.4f' % ep_r, 'CO(g): %.4f' % co,\
#      'HC(g): %.4f' % hc, 'NOX(g): %.4f' % nox, 'PMX(g): %.4f' % pmx, 'Out-in flow: %.4f' % oflow, \
#      'Bottleneck speed: %.4f' % bspeed)
# time_end=time.time()
# print('totally cost',time_end-time_start)
#
# time_start=time.time()
# vsl_controller.loadmodel()
# co = 0
# hc = 0
# nox = 0
# pmx = 0
# ep_r = 0
# oflow = 0
# bspeed = 0
# v = 29.06*np.ones(5,)
# net.start_new_simulation(write_newtrips = False)
# s, r, simulationSteps, oflow_temp, bspeed_temp, co_temp, hc_temp, nox_temp, pmx_temp = net.run_step(v)
# co = co + co_temp
# hc = hc + hc_temp
# nox = nox + nox_temp
# pmx = pmx + pmx_temp
# oflow = oflow + oflow_temp
# bspeed_temp = bspeed + bspeed_temp
# stime = np.zeros(13,)
# stime[0:12] = s
# stime[12] = 0
# while simulationSteps < 18000:
#    a = vsl_controller.choose_action(stime)
#    #a = np.clip(np.random.laplace(a, var), 0, 7.99)
#    v = from_a_to_mlv(a)
#    stime_ = np.zeros(13,)
#    s_, r, simulationSteps, oflow_temp, bspeed_temp, co_temp, hc_temp, nox_temp, pmx_temp = net.run_step(v)
#    co = co + co_temp
#    hc = hc + hc_temp
#    nox = nox + nox_temp
#    pmx = pmx + pmx_temp
#    oflow = oflow + oflow_temp
#    bspeed = bspeed + bspeed_temp
#    stime_[0:12] = s_
#    stime_[12] = simulationSteps/18000
#    stime = stime_
#    ep_r += r
# net.close()
# fname = 'output_sumo.xml'
# with open(fname, 'r') as f:  # 打开文件
#    lines = f.readlines()  # 读取所有行
#    last_line = lines[-2]  # 取最后一行
# nPos=last_line.index(traveltime)
# aat_tempo = float(last_line[nPos+16:nPos+21])
# print( 'Average Travel Time: %.4f' % aat_tempo, ' Rewards: %.4f' % ep_r, 'CO(g): %.4f' % co,\
#      'HC(g): %.4f' % hc, 'NOX(g): %.4f' % nox, 'PMX(g): %.4f' % pmx, 'Out-in flow: %.4f' % oflow, \
#      'Bottleneck speed: %.4f' % bspeed)
# time_end=time.time()
# print('totally cost',time_end-time_start)
