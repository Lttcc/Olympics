import argparse
import datetime
import os.path

from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np

from pathlib import Path
import sys
base_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_dir)


from collections import deque, namedtuple


from env.chooseenv import make
from rl_trainer.log_path import *
from rl_trainer.algo.ppo import PPO
from rl_trainer.algo.random import random_agent
import pickle
import torch.nn as nn
import torch.utils.data as Data

class Mycnn(nn.Module):
	def __init__(self):
		super(Mycnn, self).__init__()   # 继承__init__功能
		## 第一层卷积
		self.conv1 = torch.nn.Sequential(  # input_size = 25*25*1
			torch.nn.Conv2d(in_channels=1, out_channels=12, kernel_size=9, stride=1, padding=4),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(kernel_size=2, stride=2)  # output_size = 12*12*12
		)
		self.conv2 = torch.nn.Sequential(  # input_size = 12*12*12
			torch.nn.Conv2d(12, 24, 5, 1, 2),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(2, 2)  # output_size = 6*6*24
		)
		self.conv3 = torch.nn.Sequential(  # input_size = 6*6*24
			torch.nn.Conv2d(24, 48, 3, 1, 1),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(2, 2)  # output_size = 3*3*48
		)
		self.dense = torch.nn.Sequential(
			torch.nn.Linear(432, 216),
			torch.nn.ReLU(),
			torch.nn.Dropout(0.5),
			torch.nn.Linear(216, 108),
			torch.nn.ReLU(),
			torch.nn.Dropout(0.5),
			torch.nn.Linear(108, 59)
		)
	def forward(self, x):   #正向传播过程
		conv1_out = self.conv1(x.to(torch.float32))
		conv2_out = self.conv2(conv1_out)
		conv3_out = self.conv3(conv2_out)
		'''
		x.view(x.size(0), -1)的用法：
		在CNN中，因为卷积或者池化之后需要连接全连接层，所以需要把多维度的tensor展平成一维，因此用它来实现(其实就是将多维数据展平为一维数据方便后面的全连接层处理)
		'''
		res = conv3_out.view(conv3_out.size(0), -1)
		out = self.dense(res)
		return out
parser = argparse.ArgumentParser()
parser.add_argument('--game_name', default="olympics-running", type=str)
parser.add_argument('--algo', default="ppo", type=str, help="ppo/sac")
parser.add_argument('--max_episodes', default=1500, type=int)
parser.add_argument('--episode_length', default=500, type=int)
parser.add_argument('--map', default=1, type = int)
parser.add_argument('--shuffle_map', action='store_true')

parser.add_argument('--seed', default=1, type=int)

parser.add_argument("--save_interval", default=100, type=int)
parser.add_argument("--model_episode", default=0, type=int)

parser.add_argument("--load_model", action='store_true')  # 加是true；不加为false
parser.add_argument("--load_run", default=2, type=int)
parser.add_argument("--load_episode", default=900, type=int)


device = 'cpu'
RENDER = True
SAVADATA=False
actions_map = {0: [-100, -30], 1: [-100, -18], 2: [-100, -6], 3: [-100, 6], 4: [-100, 18], 5: [-100, 30], 6: [-40, -30],
               7: [-40, -18], 8: [-40, -6], 9: [-40, 6], 10: [-40, 18], 11: [-40, 30], 12: [20, -30], 13: [20, -18],
               14: [20, -6], 15: [20, 6], 16: [20, 18], 17: [20, 30], 18: [80, -30], 19: [80, -18], 20: [80, -6],
               21: [80, 6], 22: [80, 18], 23: [80, 30], 24: [140, -30], 25: [140, -18], 26: [140, -6], 27: [140, 6],
               28: [140, 18], 29: [140, 30], 30: [200, -30], 31: [200, -18], 32: [200, -6], 33: [200, 6], 34: [200, 18],
               35: [200, 30]}           #dicretise action space


def main(args):
    print("==algo: ", args.algo)
    print(f'device: {device}')
    print(f'model episode: {args.model_episode}')
    print(f'save interval: {args.save_interval}')

    env = make(args.game_name)
    if not args.shuffle_map:
        env.specify_a_map(args.map)         #specifying a map, you can also shuffle the map by not doing this step

    num_agents = env.n_player
    print(f'Total agent number: {num_agents}')

    ctrl_agent_index = 1
    print(f'Agent control by the actor: {ctrl_agent_index}')

    ctrl_agent_num = 1

    width = env.env_core.view_setting['width']+2*env.env_core.view_setting['edge']
    height = env.env_core.view_setting['height']+2*env.env_core.view_setting['edge']
    print(f'Game board width: {width}')
    print(f'Game board height: {height}')

    act_dim = env.action_dim
    obs_dim = 25*25
    print(f'action dimension: {act_dim}')
    print(f'observation dimension: {obs_dim}')

    torch.manual_seed(args.seed)
    # 定义保存路径
    run_dir, log_dir = make_logpath(args.game_name, args.algo)
    if not args.load_model:
        writer = SummaryWriter(os.path.join(str(log_dir), "{}_{} on map {}".format(
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),args.algo, 'all' if args.shuffle_map else args.map)))
        save_config(args, log_dir)

    record_win = deque(maxlen=100)
    record_win_op = deque(maxlen=100)

    if args.load_model:
        if args.algo=="ppo":
            model = PPO()
            load_dir = os.path.join(os.path.dirname(run_dir), "run" + str(args.load_run))
            model.load(load_dir,episode=args.load_episode)
        elif args.algo=="cnn":
            model=Mycnn()
            model=torch.load("./cnnmodels2.pth")
    else:
        model = PPO(run_dir)
        Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state', 'done'])

    opponent_agent = random_agent()     #we use random opponent agent here

    episode = 0
    train_count = 0
    data_temp = {'obs': [], 'action': []}
    while episode < args.max_episodes:
        state = env.reset(args.shuffle_map)    #[{'obs':[25,25], "control_player_index": 0}, {'obs':[25,25], "control_player_index": 1}]
        if RENDER:
            env.env_core.render()
        obs_ctrl_agent = np.array(state[ctrl_agent_index]['obs']).flatten()     #[25*25]
        obs_oppo_agent = state[1-ctrl_agent_index]['obs']   #[25,25]

        episode += 1
        step = 0
        Gt = 0
        '''
        ltc 把data_temp导出，data_temp只存了一个episode的数据。
        '''
        if SAVADATA and episode==1:
            imagepath = os.getcwd() + "\\" + "img"
            file1 = "data"
            file2 = "map" + str(args.map)
            fileNamePath = os.getcwd() + "\\" + file1 + "\\" + file2
            if not os.path.exists(fileNamePath):
                os.makedirs(fileNamePath)
            if not os.path.exists(imagepath):
                os.makedirs(imagepath)
            ls = os.listdir(imagepath)
            for i in ls:
                c_path = os.path.join(imagepath, i)
                os.remove(c_path)
        if SAVADATA and episode!=1:
            with open(fileNamePath+'\\' + 'data.pkl', 'wb') as f:
                pickle.dump(data_temp, f, pickle.HIGHEST_PROTOCOL)
            f.close()
            sys.exit()

        while True:
            action_opponent = opponent_agent.act(obs_oppo_agent)        #opponent action
            action_opponent = [[0],[0]]  #here we assume the opponent is not moving in the demo
            if args.algo=='ppo':
                action_ctrl_raw, action_prob= model.select_action(obs_ctrl_agent, False if args.load_model else True)
            elif args.algo=='cnn':
                obs=obs_ctrl_agent.reshape(25,25)
                obs=obs[np.newaxis,np.newaxis,:,:] #(1,1,25,25)
                obs=torch.tensor(obs)
                action_prob = model(obs)
                action_ctrl_raw=torch.max(action_prob, 1)[1].data.numpy().squeeze().item()
            '''
            ltc 存obs和ation的编号到data_temp
            '''
            if SAVADATA:
                obs=obs_ctrl_agent.reshape(25,25)#把flatten的数组重构回25*25
                data_temp['obs'].append(obs)
                data_temp['action'].append(action_ctrl_raw)

                            #inference
            action_ctrl = actions_map[action_ctrl_raw]
            action_ctrl = [[action_ctrl[0]], [action_ctrl[1]]]        #wrapping up the action

            action = [action_opponent, action_ctrl] if ctrl_agent_index == 1 else [action_ctrl, action_opponent]

            next_state, reward, done, _, info = env.step(action)

            next_obs_ctrl_agent = next_state[ctrl_agent_index]['obs']
            next_obs_oppo_agent = next_state[1-ctrl_agent_index]['obs']

            step += 1

            if not done:
                post_reward = [-1., -1.]
            else:
                if reward[0] != reward[1]:
                    post_reward = [reward[0]-100, reward[1]] if reward[0]<reward[1] else [reward[0], reward[1]-100]
                else:
                    post_reward=[-1., -1.]

            if not args.load_model:
                trans = Transition(obs_ctrl_agent, action_ctrl_raw, action_prob, post_reward[ctrl_agent_index],
                                   next_obs_ctrl_agent, done)
                model.store_transition(trans)

            obs_oppo_agent = next_obs_oppo_agent
            obs_ctrl_agent = np.array(next_obs_ctrl_agent).flatten()
            if RENDER and not SAVADATA:
                env.env_core.render()
            elif SAVADATA:
                env.env_core.renderforsave(imagepath)
            Gt += reward[ctrl_agent_index] if done else -1

            if done:
                win_is = 1 if reward[ctrl_agent_index]>reward[1-ctrl_agent_index] else 0
                win_is_op = 1 if reward[ctrl_agent_index]<reward[1-ctrl_agent_index] else 0
                record_win.append(win_is)
                record_win_op.append(win_is_op)
                print("Episode: ", episode, "controlled agent: ", ctrl_agent_index, "; Episode Return: ", Gt,
                      "; win rate(controlled & opponent): ", '%.2f' % (sum(record_win)/len(record_win)),
                      '%.2f' % (sum(record_win_op)/len(record_win_op)), '; Trained episode:', train_count)

                if not args.load_model:
                    if args.algo == 'ppo' and len(model.buffer) >= model.batch_size:
                        if win_is == 1:
                            model.update(episode)
                            train_count += 1
                        else:
                            model.clear_buffer()

                    writer.add_scalar('training Gt', Gt, episode)

                break
        if episode % args.save_interval == 0 and not args.load_model:
            model.save(run_dir, episode)





if __name__ == '__main__':
    args = parser.parse_args()
    #args.load_model = True
    #args.load_run = 3
    #args.map = 3
    #args.load_episode= 900
    main(args)