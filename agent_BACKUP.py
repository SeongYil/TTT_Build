#dqn 구현하기
# 라이브러리 불러오기
import numpy as np
import random
import copy
import datetime
import platform
import torch
import torch.nn.functional as F
from torch.serialization import load
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel\
                             import EngineConfigurationChannel


# DQN을 위한 파라미터 값 세팅 
state_size = 9
action_size = 9

load_model = True
train_mode = True

batch_size = 32
mem_maxlen = 10000
discount_factor = 0.9
learning_rate = 0.001

run_step = 100000 if train_mode else 0
test_step = 10000
#train_start_step = 500
target_update_step = 500

print_interval = 100
save_interval = 1000

epsilon_eval = 0.0
epsilon_init = 1.0 if train_mode else epsilon_eval
epsilon_min = 0.2
explore_step = run_step * 0.8
eplsilon_delta = (epsilon_init - epsilon_min)/explore_step if train_mode else 0.

VISUAL_OBS = 0
GOAL_OBS = 1
VECTOR_OBS = 2
OBS = VISUAL_OBS

# 유니티 환경 경로 
game = "TTT"
os_name = platform.system()
if os_name == 'Windows':
    env_name = f"../envs/{game}_{os_name}/{game}"
elif os_name == 'Darwin':
    env_name = f"../envs/{game}_{os_name}"


# 모델 저장 및 불러오기 경로
date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
#save_path = f"./saved_models/{game}/DQN/{date_time}"
save_path = f"./saved_models"
load_path = f"./saved_models"

# DQN 클래스 -> Deep Q Network 정의 
class DQN(torch.nn.Module):
    def __init__(self, **kwargs):
        super(DQN, self).__init__(**kwargs)
        #[1,9]
        #self.conv1 = torch.nn.Conv1d(1, 1, 9, padding=1)
        self.fc1 = torch.nn.Linear(state_size, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.q = torch.nn.Linear(32, action_size)


    def forward(self, x):
        #x = x.reshape(1,1,9)
        #x = F.relu(self.conv1(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.q(x)
        return x


# 연산 장치
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNAgent:
    def __init__(self, marker):
        self.network = DQN().to(device)
        self.target_network = copy.deepcopy(self.network)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=mem_maxlen)
        self.epsilon = epsilon_init
        self.writer = SummaryWriter() 
        load_path = f"./saved_models"

        self.marker = marker

        if(marker == 0):
            load_path = load_path + "_O"
        else:
            load_path = load_path + "_X"

        if load_model == True:
            print(f"... Load Model from {load_path}/ckpt ...")
            checkpoint = torch.load(load_path+'/ckpt', map_location=device)
            self.network.load_state_dict(checkpoint["network"])
            self.target_network.load_state_dict(checkpoint["network"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])

    # 학습 수행
    def train_model(self , state, action, reward, next_state, done):

        state, action, reward, next_state, done = map(lambda x: torch.FloatTensor(x).to(device),
                                                        [state, action, reward, next_state, done])

        #eye = torch.eye(action_size).to(device)
        #one_hot_action = eye[action.view(-1).long()]
        q = self.network(state)
        select_index = int(action)
        select_q = q[0][select_index]
        

        with torch.no_grad(): 
            target_q = self.target_network(next_state)
            target_q  = reward + torch.max(target_q) * ((1 - done) * discount_factor)

        loss = F.smooth_l1_loss(select_q, target_q[0])
        #test = torch.Tensor([[0,0.5],[0,0]]).to(device)
        #target_q = target_q.to(device=device, dtype=torch.int64)
        #loss = F.cross_entropy(test , target_q[0][argmax_index])
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 엡실론 감소
        self.epsilon = max(epsilon_min, self.epsilon - eplsilon_delta)


        return loss.item()



    # Epsilon greedy 기법에 따라 행동 결정 
    def get_action(self, state, actionmask, training=True):
        #  네트워크 모드 설정
        self.network.train(training)
        self.epsilon = self.epsilon if training else epsilon_eval

        available_action = []
        for i in range(len(actionmask[0][0])):
            if( i == 0 ):
                continue
            
            if(actionmask[0][0][i] == False):
                available_action.append(i-1)

             

        q = self.network(torch.FloatTensor(state).to(device))
        
        
        q = q.cpu().detach().numpy()

        available_q = list()

        for i in range(len(available_action)):
            i_action = available_action[i]
            available_q.append(q[0][i_action])

        available_greedy_action = np.argmax(available_q)

        # e-greedy로 행동들의 선택 확률을 계산
        pr1 = np.zeros(len(available_action))

        for i in range(len(available_action)):
            if available_q[i] == available_q[available_greedy_action]:
                pr1[i] = 1 - self.epsilon + self.epsilon/len(available_action)
            else:
                pr1[i] = self.epsilon / len(available_action)

        pr = pr1 / pr1.sum()
        
        action = np.random.choice(range(0,len(available_action)), p=pr)

        return np.array([available_action[action]]).reshape(1,1)


    # 타겟 네트워크 업데이트
    def update_target(self):
        self.target_network.load_state_dict(self.network.state_dict())

    # 네트워크 모델 저장 
    def save_model(self, in_save_path):
        print(f"... Save Model to {save_path}/ckpt ...")
        torch.save({
            "network" : self.network.state_dict(),
            "optimizer" : self.optimizer.state_dict(),
        }, in_save_path+'/ckpt')
       

    def save_onnx(self, in_save_path, state):
        print(f"... save Model from {save_path}/onnx ...")
        torch.onnx.export(
            self.network,
            state,
            in_save_path+ f'/{self.marker}.onnx',
            export_params=True, 
            opset_version=9, 
            do_constant_folding=True,
            )


    # 학습 기록 
    def write_summray(self, score, loss, epsilon, step):
            self.writer.add_scalar("run/score", score, step)
            self.writer.add_scalar("model/loss", loss, step)
            self.writer.add_scalar("model/epsilon", epsilon, step)

# Main 함수 -> 전체적으로 DQN 알고리즘을 진행 
if __name__ == '__main__':
    # 유니티 환경 경로 설정 (file_name)
    engine_configuration_channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name="./TTT",
                           side_channels=[engine_configuration_channel],
                           no_graphics=True
                           )
    env.reset()

    # 유니티 브레인 설정 
    behavior_name = list(env.behavior_specs.keys())
    engine_configuration_channel.set_configuration_parameters(time_scale=1000.0)


    #spec = env.behavior_specs[behavior_name[0]]
    #dec1, term1 = env.get_steps(behavior_name[1])

    # DQNAgent 클래스를 agent로 정의 
    O_agent = DQNAgent(0)
    X_agent = DQNAgent(1)


    


    
    
    episode = 0
    O_score = 0
    X_score = 0 
    O_losses, O_scores = [], []
    X_losses, X_scores = [], []
    for step in range(run_step + test_step):
        if step == run_step:
            if train_mode:
                O_agent.save_model(save_path + "_O")
                X_agent.save_model(save_path + "_X")
            print("TEST START")
            train_mode = False
            engine_configuration_channel.set_configuration_parameters(time_scale=1.0)

        dec_o, term_o = env.get_steps(behavior_name[1])
        o_state = dec_o.obs[0]

        #state = torch.FloatTensor(o_state).to(device)

        #O_agent.save_onnx(save_path + "_O", state)


        o_action = O_agent.get_action(o_state,dec_o.action_mask, train_mode)
        o_real_action = o_action + 1
        action_tuple = ActionTuple()
        action_tuple.add_discrete(o_real_action)
        env.set_actions(behavior_name[1], action_tuple)
        env.step()

        dec, term = env.get_steps(behavior_name[1])
        done = len(term.agent_id) > 0

        reward = term.reward if done else dec.reward
        next_state = None

        if( done ):
            next_state = term.obs[0]
            dec_x, term_x = env.get_steps(behavior_name[0])
            X_score += term_x.reward[0]

        else:
            next_state = dec.obs[0]

        O_score += reward[0]

        #dec_x, term_x = env.get_steps(behavior_name[0])
        #state_x = dec_x.obs[0]

        if train_mode:# and step > max(batch_size, train_start_step):
            # 학습 수행
            loss = O_agent.train_model(o_state, o_action, reward, next_state, [done])
            O_losses.append(loss)

            # 타겟 네트워크 업데이트 
            if step % target_update_step == 0:
                O_agent.update_target()
                X_agent.update_target()


        if done:
            episode +=1
            O_scores.append(O_score)
            O_score = 0

            X_scores.append(X_score)
            X_score = 0

            # 게임 진행 상황 출력 및 텐서 보드에 보상과 손실함수 값 기록 
            if episode % print_interval == 0:
                O_mean_score = np.mean(O_scores)
                O_mean_loss = np.mean(O_losses)
                O_agent.write_summray(O_mean_score, O_mean_loss, O_agent.epsilon, step)
                O_losses, O_scores = [], []

                X_mean_score = np.mean(X_scores)
                X_mean_loss = np.mean(X_losses)
                X_agent.write_summray(X_mean_score, X_mean_loss, X_agent.epsilon, step)
                X_losses, X_scores = [], []




                print(f"{episode} Episode / Step: {step} / O_Score: {O_mean_score:.2f} / " +\
                      f"Loss: {O_mean_loss:.4f} / Epsilon: {O_agent.epsilon:.4f}")

                print(f"{episode} Episode / Step: {step} / X_Score: {X_mean_score:.2f} / " +\
                      f"Loss: {X_mean_loss:.4f} / Epsilon: {X_agent.epsilon:.4f}")
            env.step()
            continue

            ## 네트워크 모델 저장 
            #if train_mode and episode % save_interval == 0:
            #    agent.save_model()

        x_dec, x_term = env.get_steps(behavior_name[0])
        x_state = x_dec.obs[0] 
        x_action = X_agent.get_action(x_state, x_dec.action_mask , train_mode)
        x_real_action = x_action + 1
        action_tuple = ActionTuple()
        action_tuple.add_discrete(x_real_action)
        env.set_actions(behavior_name[0], action_tuple)
        env.step()

        dec, term = env.get_steps(behavior_name[0])
        done = len(term.agent_id) > 0

        reward = term.reward if done else dec.reward
        next_state = None

        if( done ):
            next_state = term.obs[0]
            dec_o, term_o = env.get_steps(behavior_name[1])
            O_score += term_o.reward[0]
        else:
            next_state = dec.obs[0]

        X_score += reward[0]
        
        #dec_x, term_x = env.get_steps(behavior_name[0])
        #state_x = dec_x.obs[0]

        if train_mode:# and step > max(batch_size, train_start_step):
            # 학습 수행
            loss = X_agent.train_model(x_state, x_action, reward, next_state, [done])
            X_losses.append(loss)

            # 타겟 네트워크 업데이트 
            if step % target_update_step == 0:
                O_agent.update_target()
                X_agent.update_target()


        if done:
            episode +=1
            O_scores.append(O_score)
            O_score = 0

            X_scores.append(X_score)
            X_score = 0

            # 게임 진행 상황 출력 및 텐서 보드에 보상과 손실함수 값 기록 
            if episode % print_interval == 0:
                O_mean_score = np.mean(O_scores)
                O_mean_loss = np.mean(O_losses)
                O_agent.write_summray(O_mean_score, O_mean_loss, O_agent.epsilon, step)
                O_losses, O_scores = [], []

                X_mean_score = np.mean(X_scores)
                X_mean_loss = np.mean(X_losses)
                X_agent.write_summray(X_mean_score, X_mean_loss, X_agent.epsilon, step)
                X_losses, X_scores = [], [] 

                            
                print(f"{episode} Episode / Step: {step} / O_Score: {O_mean_score:.2f} / " +\
                      f"Loss: {O_mean_loss:.4f} / Epsilon: {O_agent.epsilon:.4f}")

                print(f"{episode} Episode / Step: {step} / X_Score: {X_mean_score:.2f} / " +\
                      f"Loss: {X_mean_loss:.4f} / Epsilon: {X_agent.epsilon:.4f}")

                            



            env.step()



    env.close()


