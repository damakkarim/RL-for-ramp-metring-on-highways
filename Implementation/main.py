import os
import sys

import traci
from Environment import SumoEnvironment
from Agent import DQNAgent


if "SUMO_HOME" not in os.environ:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

# Initialize the environment and agent
env = SumoEnvironment(r'C:\Users\Siwar\Desktop\girl\RL-for-ramp-metring-on-highways\SUMO architecture/light1.sumocfg')
agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)
average_wait_time=0
# run the simulation please
def run_simulation():
    traci.start(['sumo-gui', '-c', env.sumo_cfg_file])
    total_wait_time = 0
    total_vehicles_passed = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        state = env.get_state()
        action = agent.choose_action(state)
        env.apply_action(action)
        reward, wait_time, vehicles_passed = env.step(action)
        total_wait_time += wait_time
        total_vehicles_passed += vehicles_passed
        next_state, reward, done = env.step(action)

        # Print  les etats les  actions et les rewads
        print(f"State: {state}, Action: {action}, Reward: {reward}")
    average_wait_time = total_wait_time / total_vehicles_passed if total_vehicles_passed else 0

    traci.close()

# Main execution
if __name__ == '__main__':
    try:
        run_simulation()
        print("average_wait_time",average_wait_time)


    except traci.exceptions.FatalTraCIError as e:
        print(f"Error during simulation: {e}")