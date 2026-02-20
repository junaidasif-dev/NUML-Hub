import gymnasium as gym

# ----------------------------
# Environment 1: CartPole-v1
# ----------------------------

# Step 1: Create the environment (render_mode="human" opens a visual window)
env = gym.make("CartPole-v1", render_mode="human")

# Step 2: Print basic environment info
print("Environment Name  :", "CartPole-v1")
print("Observation Space :", env.observation_space)   # what the agent sees
print("Action Space      :", env.action_space)         # 0 = push left, 1 = push right

# Step 3: Reset the environment to get starting state
observation, info = env.reset()

print("\nStarting Observation:", observation)
print("  Cart Position :", observation[0])
print("  Cart Velocity :", observation[1])
print("  Pole Angle    :", observation[2])
print("  Pole Velocity :", observation[3])
print("Info            :", info)

# Step 4: Run 5 steps with random actions
print("\n--- Running 50 Steps ---")

for step in range(50):

    # Pick a random action
    action = env.action_space.sample()

    # Take the action
    observation, reward, terminated, truncated, info = env.step(action)

    # Is the episode over?
    done = terminated or truncated

    # Print all variables
    print(f"\nStep      : {step + 1}")
    print(f"Action    : {action}  (0=Left, 1=Right)")
    print(f"Obs[0] Cart Position : {observation[0]:.4f}")
    print(f"Obs[1] Cart Velocity : {observation[1]:.4f}")
    print(f"Obs[2] Pole Angle    : {observation[2]:.4f}")
    print(f"Obs[3] Pole Velocity : {observation[3]:.4f}")
    print(f"Reward    : {reward}")
    print(f"Terminated: {terminated}")
    print(f"Truncated : {truncated}")
    print(f"Done      : {done}")
    print(f"Info      : {info}")

    if done:
        observation, info = env.reset()

# Step 5: Close the environment
env.close()
print("\nDone!")
