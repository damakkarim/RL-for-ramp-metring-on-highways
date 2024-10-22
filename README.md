🚦 RLOC Project 2024-2025: Reinforcement Learning for Highway Ramp Metering
📋 Project Description
This project applies Q-learning and Deep Q-learning algorithms to optimize the control of a highway entrance ramp managed by a traffic light. The goal is to improve the traffic flow on both the highway and the ramp using the SUMO (Simulation of Urban Mobility) traffic simulator.

🎯 Objectives
Model the state of traffic on the highway and ramp.
Control the traffic light on the ramp to optimize traffic flow using Q-learning.
Test various scenarios by adjusting parameters such as the highway length, initial vehicle density, and traffic flow.
🛠️ Technologies Used
🔧 SUMO for traffic simulation.
🧠 Q-learning and Deep Q-learning for intelligent traffic light control.
📊 Python for implementing and simulating the learning algorithms.


📂 Project Structure


📌 State Variables
State variables represent the traffic conditions on the highway and ramp, including parameters like vehicle speed and position.

🔄 Action Variables
The algorithm controls the green light proportion in a given cycle, affecting the vehicle flow at the ramp.

🏆 Reward
The reward model reflects the optimization of overall traffic, considering both highway and ramp efficiency.

🔗 Useful Links
DQN Traffic Control Framework
Arxiv Article
🧪 Tested Scenarios
We varied several parameters to simulate different scenarios:

Highway length 🚗
Speed limits 🏎️
Initial vehicle density 🚙
Vehicle flow from the ramp 🚦


👨‍💻 Authors
Project supervised by Nadir Farhi with contributions from Romain Ducrocq.

