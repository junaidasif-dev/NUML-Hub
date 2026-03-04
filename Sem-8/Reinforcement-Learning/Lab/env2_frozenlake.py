import gymnasium as gym

# ----------------------------
# Environment 2: FrozenLake-v1
# ----------------------------

# Step 1: Create the environment (render_mode="human" opens a visual window)
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")

# Step 2: Print basic environment info
print("Environment Name  :", "FrozenLake-v1")
print("Observation Space :", env.observation_space)   # 16 tiles (4x4 grid)
print("Action Space      :", env.action_space)         # 0=Left 1=Down 2=Right 3=Up
print("Number of States  :", env.observation_space.n)
print("Number of Actions :", env.action_space.n)

# Step 3: Reset the environment
observation, info = env.reset()

print("\nStarting Observation (tile number):", observation)
print("Info :", info)
print("  prob = probability of this transition:", info.get("prob", "N/A"))

# Step 4: Run 5 steps with random actions
print("\n--- Running 50 Steps ---")

action_names = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}

for step in range(50):

    # Pick a random action
    action = env.action_space.sample()

    # Take the action
    observation, reward, terminated, truncated, info = env.step(action)

    # Is the episode over?
    done = terminated or truncated

    # Print all variables
    print(f"\nStep        : {step + 1}")
    print(f"Action      : {action}  ({action_names[action]})")
    print(f"Observation : {observation}  (current tile on grid)")
    print(f"Reward      : {reward}  (1 = reached goal, 0 = otherwise)")
    print(f"Terminated  : {terminated}")
    print(f"Truncated   : {truncated}")
    print(f"Done        : {done}")
    print(f"Info        : {info}")

    if done:
        print("Episode ended. Resetting...")
        observation, info = env.reset()

# Step 5: Close the environment
env.close()
print("\nDone!")
