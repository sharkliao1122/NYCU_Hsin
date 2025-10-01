import numpy as np
import random

# Q-learning 參數
alpha = 0.1      # 學習率
gamma = 0.9      # 折扣因子
epsilon = 0.2    # 探索率
num_episodes = 7000
max_steps = 5

# 狀態空間：low, high (1~1000)
state_space = [(low, high) for low in range(1, 1001) for high in range(low, 1001)]
# 行動空間：猜 low~high 之間的數字

def get_state(low, high):
    return state_space.index((low, high))

def choose_action(q_table, state, low, high):
    if np.random.rand() < epsilon:
        return random.randint(low, high)
    else:
        actions = [q_table[state, a-1] for a in range(low, high+1)]
        return np.argmax(actions) + low

def update_range(guess, secret, low, high):
    if guess < secret:
        low = max(low, guess+1)
    elif guess > secret:
        high = min(high, guess-1)
    return low, high

q_table = np.zeros((len(state_space), 1000))

for episode in range(num_episodes):
    secret = random.randint(1, 100)
    low, high = 1, 100
    state = get_state(low, high)
    for step in range(max_steps):
        action = choose_action(q_table, state, low, high)
        reward = 0
        done = False
        if action == secret:
            reward = 10
            done = True
        else:
            reward = -1
        next_low, next_high = update_range(action, secret, low, high)
        next_state = get_state(next_low, next_high)
        q_table[state, action-1] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action-1])
        low, high = next_low, next_high
        state = next_state
        if done:
            break

# 測試 agent
success = 0
for test in range(100):
    secret = random.randint(1, 10)
    low, high = 1, 10
    state = get_state(low, high)
    for step in range(max_steps):
        action = choose_action(q_table, state, low, high)
        if action == secret:
            success += 1
            break
        low, high = update_range(action, secret, low, high)
        state = get_state(low, high)
print(f"Q-learning agent 猜中率: {success}% (100次測試)")
