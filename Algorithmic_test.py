import gym

go_env = gym.make('gym_go:go-v0', size=5, reward_method='real')

first_action = (2,3)
second_action = (3,4)
state, reward, done, info = go_env.step(first_action)

print(state)
print(reward)
print(done)
print(info)

# go_env.render('terminal')

state, reward, done, info = go_env.step(None)
# go_env.render('terminal')

print(state)
print(reward)
print(done)
print(info)

state, reward, done, info = go_env.step(None)
# go_env.render('terminal')

print(state)
print(reward)
print(done)
print(info)
