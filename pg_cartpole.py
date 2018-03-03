import tensorflow as tf
import numpy as np
import gym

def mlp(x, hidden_sizes=(32,32), activation=tf.tanh):
    for size in hidden_sizes:
        x = tf.layers.dense(x, units=size, activation=activation)
    return x

def discount_cumsum(x, gamma):
    n = len(x)
    x = np.array(x)
    y = gamma**np.arange(n)
    z = np.zeros_like(x, dtype=np.float32)
    for j in range(n):
        z[j] = sum(x[j:] * y[:n-j])
    return z

def train(env_name='CartPole-v0', hidden_dim=32, n_layers=1,
          lr=1e-2, gamma=0.99, n_iters=50, batch_size=5000
          ):

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # make model
    with tf.variable_scope('model'):
        obs_ph = tf.placeholder(shape=(None, obs_dim), dtype=tf.float32)
        net = mlp(obs_ph, hidden_sizes=[hidden_dim]*n_layers)
        logits = tf.layers.dense(net, units=n_acts, activation=None)
        actions = tf.squeeze(tf.multinomial(logits=logits,num_samples=1), axis=1)

    # make loss
    adv_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
    act_ph = tf.placeholder(shape=(None,), dtype=tf.int32)
    action_one_hots = tf.one_hot(act_ph, n_acts)
    log_probs = tf.reduce_sum(action_one_hots * tf.nn.log_softmax(logits), axis=1)
    loss = -tf.reduce_mean(adv_ph * log_probs)

    # make train op
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # train model
    def train_one_iteration():
        batch_obs, batch_acts, batch_rtgs, batch_rets, batch_lens = [], [], [], [], []

        obs, rew, done, ep_rews = env.reset(), 0, False, []
        while True:
            batch_obs.append(obs.copy())
            act = sess.run(actions, {obs_ph: obs.reshape(1,-1)})[0]
            obs, rew, done, _ = env.step(act)
            batch_acts.append(act)
            ep_rews.append(rew)
            if done:
                batch_rets.append(sum(ep_rews))
                batch_lens.append(len(ep_rews))
                batch_rtgs += list(discount_cumsum(ep_rews, gamma))
                obs, rew, done, ep_rews = env.reset(), 0, False, []
                if len(batch_obs) > batch_size:
                    break

        # normalize advs trick:
        batch_advs = np.array(batch_rtgs)
        batch_advs = (batch_advs - np.mean(batch_advs))/(np.std(batch_advs) + 1e-8)
        batch_loss, _ = sess.run([loss, train_op], feed_dict={obs_ph: np.array(batch_obs),
                                                              act_ph: np.array(batch_acts),
                                                              adv_ph: batch_advs})
        return batch_loss, batch_rets, batch_lens

    for i in range(n_iters):
        batch_loss, batch_rets, batch_lens = train_one_iteration()
        print('itr: %d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

if __name__ == '__main__':
    train()