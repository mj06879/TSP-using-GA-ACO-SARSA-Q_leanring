# A Comparative Analysis of Genetic Algorithms, Ant Colony Optimization, Q-Learning and SARSA for Solving the Travelling Salesman Problem

## Overview

This repository presents a comprehensive comparative analysis of four optimization techniques: Genetic Algorithms (GA), Ant Colony Optimization (ACO), Q-Learning, and State-Action-Reward-State-Action (SARSA) in solving the Travelling Salesman Problem (TSP). Each algorithm's effectiveness, efficiency, and suitability for addressing the TSP are meticulously evaluated and benchmarked against one another.

## Introduction

The Travelling Salesman Problem (TSP) is a classic optimization challenge that seeks the most efficient route for a salesman to visit a given set of cities exactly once and then return to the origin city. Given its combinatorial nature, TSP has been the subject of extensive research, leading to the development of various heuristic and metaheuristic algorithms like GA, ACO, Q-Learning, and SARSA.

## Objective

The primary objective of this repository is to:

- Compare the efficiency, effectiveness, and computational complexity of GA, ACO, Q-Learning, and SARSA in solving TSP.
- Provide empirical evidence through experimentation and analysis.

## Algorithms Explored

### Genetic Algorithms (GA)

- **Description**: A genetic algorithm mimics the process of natural evolution to generate high-quality solutions.
- **Advantages**: Exploration of solution space, parallelism, and robustness.
- **Applications**: TSP, scheduling, optimization problems.

### Ant Colony Optimization (ACO)

- **Description**: ACO is inspired by the foraging behaviour of ants to find optimal paths.
- **Advantages**: Adaptability, ability to find near-optimal solutions, and scalability.
- **Applications**: Routing, TSP, network design.

### Q-Learning

- **Description**: A model-free reinforcement learning algorithm that learns optimal actions by exploring the state-action space.
- **Advantages**: Learning from rewards, no need for explicit models, and adaptability.
- **Applications**: Game playing, robotics, optimization problems.

### SARSA (State-Action-Reward-State-Action)

- **Description**: SARSA is an on-policy temporal difference control algorithm that estimates the value of state-action pairs.
- **Advantages**: Convergence guarantees, efficiency with limited computational resources.
- **Applications**: Reinforcement learning tasks, optimization problems, game playing.

## Methodology

1. **Dataset**: Utilization of standard TSP datasets to ensure consistency and comparability.
2. **Experiment Design**: Comprehensive experimental setup with controlled variables and parameters.
3. **Performance Metrics**: Evaluation based on solution quality, computational time, convergence rate, and scalability.

To have a better understanding and clarification, refer to the report link [here](https://github.com/mj06879/TSP-using-GA-ACO-SARSA-Q_leanring/blob/main/AI_Project_Report_mj06879_sa06840.pdf)
## Results

To ensure a fair comparison, all four algorithms were run on the same dataset consisting of 194 cities. Firstly, each algorithm was analysed individually using different parameter values to check which set of parameters gave the best results for the specific algorithm. Once this was determined, the algorithms were run together on a set number of epochs (4000) and each algorithm was run using the set of parameters from which got the best results.

<img width="659" alt="image" src="https://github.com/mj06879/TSP-using-GA-ACO-SARSA-Q_leanring/assets/78081958/e21d961e-8f08-4f99-8e8d-0e2cd8dad5b3">
<img width="531" alt="image" src="https://github.com/mj06879/TSP-using-GA-ACO-SARSA-Q_leanring/assets/78081958/6f7bd6b2-615a-4e71-a722-bc90f390646f">
<img width="538" alt="image" src="https://github.com/mj06879/TSP-using-GA-ACO-SARSA-Q_leanring/assets/78081958/d786f442-694d-474c-945e-a0008c55e65e">
<img width="565" alt="image" src="https://github.com/mj06879/TSP-using-GA-ACO-SARSA-Q_leanring/assets/78081958/62f03470-6ec6-4f35-9d45-6efae97f5746">
<img width="564" alt="image" src="https://github.com/mj06879/TSP-using-GA-ACO-SARSA-Q_leanring/assets/78081958/ae2c230d-9818-490c-836a-e6d5c678f219">

Now for a complete and fair comparison, we took the best parameters for each algorithm

<img width="602" alt="image" src="https://github.com/mj06879/TSP-using-GA-ACO-SARSA-Q_leanring/assets/78081958/630a954b-4acf-43f9-aa68-a4fe5be3b99d">
<img width="573" alt="image" src="https://github.com/mj06879/TSP-using-GA-ACO-SARSA-Q_leanring/assets/78081958/6ceab265-0509-44a3-8cf4-a572aaf597ba">

According to our reflections, Ant Colony Optimisation has the best performance due to its natural pheromone-based approach that is well suited for optimisation problems specifically the TSP whereas Genetic Algorithm is a general optimisation approach which needs to be altered to solve the TSP. On the other hand, reinforcement learning techniques are generally sequential decision-making techniques that are not made for combinatorial optimisation problems such as the TSP. Moreover, ACO converges relatively quickly because ants communicate with each other and thus build upon each other’s tours, reinforcing the best tours whereas genetic algorithms can be slow to converge due to the random nature of their crossover and mutation operations.

## Conclusion

In conclusion, we studied the theoretical implementations of four different AI-based techniques: Genetic Algorithms, Ant Colony Optimisation, Q-Learning and SARSA performed a comparative analysis of all four by using their 
implementations to solve the Travelling Salesman Problem which is an NP-hard problem. The comparative analysis was done in two parts. Firstly, each algorithm was analysed individually using different parameter values to check which set of parameters gave the best results for the specific algorithm. Secondly, the algorithms were run together on a set number of epochs and each algorithm was run using the set of parameters from which it got the best results. 

According to our results, Ant Colony Optimisation was able to achieve the most optimal solution in the least number of iterations followed by SARSA Learning which took a considerably greater number of iterations. Genetic algorithm was also able to give a good optimal value however, it converged late as compared to the other algorithms. Lastly, Q learning gave the highest value (least optimal) however, it converges in fewer iterations as compared to Genetic Algorithms. This study can further be improved by implementing and analysing hybrid techniques between various Evolutionary Algorithms and Reinforcement Learning techniques


## Usage

To run the experiments separately on customized parameters, run the files `ACO_model.py`, `RL_models.py` & `TSP_Genetic_solver.py`
To see the comparative analysis and graphs, run the file `plot.py`

## License

This project is licensed under the [MIT License](LICENSE).
