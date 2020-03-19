# Written by Jingwei Wan

import gym

BOARD_SIZE = 5

go_env = gym.make('gym_go:go-v0', size=BOARD_SIZE, reward_method='real')

initial_state = go_env.reset()

first_action = (2,3)
second_action = (3,4)
state, reward, done, info = go_env.step_batch(initial_state, first_action)

print(state[0])
print(state[1])
print(reward)
print(done)
print(info)

state, reward, done, info = go_env.step_batch(initial_state, first_action)
print(state[0])
print(state[1])
print(reward)
print(done)
print(info)