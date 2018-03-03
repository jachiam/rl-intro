import tensorflow as tf
import numpy as np
import gym

class ReplayBuffer:

    def __init__(self, obs_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros(size, dtype=np.int32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.max_size = size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])


def mlp(x, hidden_sizes=(32,32), activation=tf.tanh):
    for size in hidden_sizes:
        x = tf.layers.dense(x, units=size, activation=activation)
    return x


def train(env_name='CartPole-v0', hidden_dim=32, n_layers=1,
          lr=1e-3, gamma=0.99, n_epochs=50, steps_per_epoch=5000, 
          batch_size=32, target_update_freq=2500, final_epsilon=0.05,
          finish_decay=50000, replay_size=5000, steps_before_training=1500
          ):

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    replay_buffer = ReplayBuffer(obs_dim=obs_dim, size=replay_size)

    # make model
    with tf.variable_scope('main'):
        obs_ph = tf.placeholder(shape=(None, obs_dim), dtype=tf.float32)
        net = mlp(obs_ph, hidden_sizes=[hidden_dim]*n_layers)
        q_vals = tf.layers.dense(net, units=n_acts, activation=None)

    with tf.variable_scope('target'):
        obs_targ_ph = tf.placeholder(shape=(None, obs_dim), dtype=tf.float32)
        net = mlp(obs_targ_ph, hidden_sizes=[hidden_dim]*n_layers)
        q_targ = tf.layers.dense(net, units=n_acts, activation=None)

    # make loss
    act_ph = tf.placeholder(shape=(None,), dtype=tf.int32)
    rew_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
    done_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
    action_one_hots = tf.one_hot(act_ph, n_acts)
    q_a = tf.reduce_sum(action_one_hots * q_vals, axis=1)
    target = rew_ph + gamma * (1 - done_ph) * tf.stop_gradient(tf.reduce_max(q_targ, axis=1))
    loss = tf.reduce_mean((q_a - target)**2)

    main_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='main')
    target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target')
    assign_ops = [tf.assign(target_var, main_var) for target_var, main_var in zip(target_vars, main_vars)]
    target_update_op = tf.group(*assign_ops)

    # make train op
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    obs, rew, done = env.reset(), 0, False
    epsilon = 1
    ep_ret, ep_len = 0, 0
    epoch_losses, epoch_rets, epoch_lens = [], [], []
    for t in range(n_epochs * steps_per_epoch):
        if np.random.rand() < epsilon:
            act = np.random.randint(n_acts)
        else:
            cur_q = sess.run(q_vals, feed_dict={obs_ph: obs.reshape(1,-1)})
            act = np.argmax(cur_q)
        next_obs, rew, done, _ = env.step(act)
        replay_buffer.store(obs, act, rew, next_obs, done)

        ep_ret += rew
        ep_len += 1

        if done:
            obs, rew, done = env.reset(), 0, False
            epoch_rets.append(ep_ret)
            epoch_lens.append(ep_len)
            ep_ret = 0
            ep_len = 0

        if t > steps_before_training:
            batch = replay_buffer.sample_batch(batch_size)
            step_loss, _ = sess.run([loss, train_op], feed_dict={obs_ph: batch['obs1'],
                                                                 obs_targ_ph: batch['obs2'],
                                                                 act_ph: batch['acts'],
                                                                 rew_ph: batch['rews'],
                                                                 done_ph: batch['done']
                                                                 })
            epoch_losses.append(step_loss)

        if t % target_update_freq == 0:
            sess.run(target_update_op)

        epsilon = 1 + (final_epsilon - 1)*min(1, t/finish_decay)

        if t % steps_per_epoch == 0 and t>0:
            epoch = t // steps_per_epoch
            print('epoch: %d \t loss: %.3f \t ret: %.3f \t len: %.3f \t epsilon: %.3f'%
                    (epoch, np.mean(epoch_losses), np.mean(epoch_rets), np.mean(epoch_lens), epsilon))
            epoch_losses = []
            epoch_rets = []
            epoch_losses = []

if __name__ == '__main__':
    train()