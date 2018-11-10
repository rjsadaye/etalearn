from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import gym, logging
from baselines import logger
import sys
import tensorflow as tf

# env_id, num_timesteps, seed, num_options,app, saves ,wsaves, epoch,dc
def enjoy(env_id, num_timesteps, seed, num_options,dc):
    '''
    from baselines.ppo1 import mlp_policy, pposgd_simple
    sess = U.make_session(num_cpu=1)
    sess.__enter__()
    logger.session().__enter__()
    set_global_seeds(seed)
    env = gym.make(env_id)
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    env = bench.Monitor(env, osp.join(logger.get_dir(), "monitor.json"))
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    '''
    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)
    env = gym.make(env_id)
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2, num_options=num_options, dc=dc)
    env = bench.Monitor(env, logger.get_dir() and 
        osp.join(logger.get_dir(), "monitor.json"))
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    pi = policy_fn('pi', env.observation_space, env.action_space)
    #saver = tf.train.Saver()
    #with tf.Session() as ses:
    sess = tf.Session()
    saver = tf.train.import_meta_graph('hopperseed777savename_2opts_saves/hopperseed777savename_epoch_485.ckpt.meta')
    saver.restore(sess, 'hopperseed777savename_2opts_saves/hopperseed777savename_epoch_485.ckpt')#.data-00000-of-00001')
    #done = False
    #while not done:
    #action = pi.act(True, obs)[0]
    #obs, reward, done, info = env.step(action)
    done = False
    obs = env.reset()
    option = pi.get_option(obs)

    while not done:
        action = pi.act(True, obs, pi.option)[0]
        #obs, reward, done, info =
        env.step(action)
        env.render()
    #act
    #env.render()
    #obs, reward, done, info = env.step(action)
    #action = pi.act(True, obs)[0]
    #env.render()


def main():
    enjoy('Hopper-v2', num_timesteps=1e6, seed=777, num_options=2,dc=0.1)


if __name__ == '__main__':
    main()
