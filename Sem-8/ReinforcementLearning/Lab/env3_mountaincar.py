import gymnasium as gym

# ----------------------------
# Environment 3: MountainCar-v0
# ----------------------------

# Step 1: Create the environment (render_mode="human" opens a visual window)
env = gym.make("MountainCar-v0", render_mode="human")

# Step 2: Print basic environment info
print("Environment Name  :", "MountainCar-v0")
print("Observation Space :", env.observation_space)   # position and velocity
print("Action Space      :", env.action_space)         # 0=Left 1=Nothing 2=Right
print("Position Range    : min =", env.observation_space.low[0],  "max =", env.observation_space.high[0])
print("Velocity Range    : min =", env.observation_space.low[1],  "max =", env.observation_space.high[1])

# Step 3: Reset the environment
observation, info = env.reset()

print("\nStarting Observation:", observation)
print("  Position :", observation[0])
print("  Velocity :", observation[1])
print("Info       :", info)

# Step 4: Run 5 steps with random actions
print("\n--- Running 50 Steps ---")

action_names = {0: "PUSH LEFT", 1: "NO PUSH", 2: "PUSH RIGHT"}

for step in range(50):

    # Pick a random action
    action = env.action_space.sample()

    # Take the action
    observation, reward, terminated, truncated, info = env.step(action)

    # Is the episode over?
    done = terminated or truncated

    # Print all variables
    print(f"\nStep       : {step + 1}")
    print(f"Action     : {action}  ({action_names[action]})")
    print(f"Position   : {observation[0]:.4f}  (car position on the hill)")
    print(f"Velocity   : {observation[1]:.4f}  (car speed)")
    print(f"Reward     : {reward}  (-1 every step until goal reached)")
    print(f"Terminated : {terminated}")
    print(f"Truncated  : {truncated}")
    print(f"Done       : {done}")
    print(f"Info       : {info}")

    if done:
        print("Episode ended. Resetting...")
        observation, info = env.reset()

# Step 5: Close the environment
env.close()
print("\nDone!")
