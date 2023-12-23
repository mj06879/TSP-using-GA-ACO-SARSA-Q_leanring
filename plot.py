# epsilon = high, low, mid, decay
# learning_rate = high, low, mid
# discount = high, low , mid

# avg reward vs no of training episodes

import matplotlib.pyplot as plt
import copy 
import time
from src.RL_models import RL_Methods_Analysis
from src.TSP_Genetic_solver import Genetic_Analysis
from src.ACO_model import ACO_Analysis
# from SarsaLearning import SarsaLearning

class AnalysisPlot():
    def __init__(self) -> None:
        self.epsilon = [0.1,0.01]
        self.learning_rate = [0.1,0.5,0.9]
        self.epoch = 4000
        self.mutation_rate = [0.1, 0.3, 0.7]
        self.population_size = [10, 30]
        self.alpha = [1, 3]
        self.beta = [10, 7]

        
    def Q_learning(self):      # works for 1 datasetonly. In utils.py, keep only one instance
        
        n = 0
        results= []
        for ep in self.epsilon: # len = 2
            for lr in self.learning_rate: # len = 3
                if n==0:
                    values, true_best, tag = RL_Methods_Analysis(epsilon=ep, gamma=0.9, learning_rate=lr,epochs=self.epoch, method="Q")
                else: 
                    values,_, _ = RL_Methods_Analysis(epsilon=ep, gamma=0.9, learning_rate=lr,epochs=self.epoch, method="Q")
                
                print(len(values))
                results.append((values, ep, lr))
        x_value = [_ for _ in range(self.epoch)]
        # print(results)
        
        plt.figure(figsize=(19, 7))
        plt.plot(x_value, results[0][0],label = "$\\epsilon = 0.1$, $\\alpha = 0.1$" + ", best values = " + str(round(results[0][0][-1], 2)), linestyle = "-")
        plt.plot(x_value, results[1][0],label = "$\\epsilon = 0.1$, $\\alpha = 0.5$" + ", best values = " + str(round(results[1][0][-1], 2)), linestyle = "-")
        plt.plot(x_value, results[2][0],label = "$\\epsilon = 0.1$, $\\alpha = 0.9$" + ", best values = " + str(round(results[2][0][-1], 2)), linestyle = "--")
        plt.plot(x_value, results[3][0],label = "$\\epsilon = 0.01$, $\\alpha = 0.1$" + ", best values = " + str(round(results[3][0][-1], 2)), linestyle = "--")
        plt.plot(x_value, results[4][0],label = "$\\epsilon = 0.1$, $\\alpha = 0.5$" + ", best values = " + str(round(results[4][0][-1], 2)), linestyle = "--")
        plt.plot(x_value, results[5][0],label = "$\\epsilon = 0.1$, $\\alpha = 0.9$" + ", best values = " + str(round(results[5][0][-1], 2)), linestyle = "--")
        plt.hlines(true_best, xmin=0, xmax=self.epoch, color="r", label="True best, " + str(true_best))
        plt.title("Q-Learning Analysis Plot")   
        plt.ylabel("Tour Distance")
        plt.xlabel("Number of Epocs")
        plt.legend()
        plt.show()
        # plt.savefig(f"./figures/Distance_evolution_{tag}")

    def SARSA_learning(self):      # works for 1 datasetonly. In utils.py, keep only one instance
        
        n = 0
        results= []
        for ep in self.epsilon: # len = 2
            for lr in self.learning_rate: # len = 3
                if n==0:
                    values, true_best, tag = RL_Methods_Analysis(epsilon=ep, gamma=0.9, learning_rate=lr,epochs=self.epoch, method="SARSA")
                else: 
                    values,_, _ = RL_Methods_Analysis(epsilon=ep, gamma=0.9, learning_rate=lr,epochs=self.epoch, method="SARSA")
                
                print(len(values))
                results.append((values, ep, lr))
        x_value = [_ for _ in range(self.epoch)]
        # print(results)
        
        plt.figure(figsize=(19, 7))
        plt.plot(x_value, results[0][0],label = "$\\epsilon = 0.1$, $\\alpha = 0.1$" + ", best values = " + str(round(results[0][0][-1], 2)), linestyle = "-")
        plt.plot(x_value, results[1][0],label = "$\\epsilon = 0.1$, $\\alpha = 0.5$" + ", best values = " + str(round(results[1][0][-1], 2)), linestyle = "-")
        plt.plot(x_value, results[2][0],label = "$\\epsilon = 0.1$, $\\alpha = 0.9$" + ", best values = " + str(round(results[2][0][-1], 2)), linestyle = "--")
        plt.plot(x_value, results[3][0],label = "$\\epsilon = 0.01$, $\\alpha = 0.1$" + ", best values = " + str(round(results[3][0][-1], 2)), linestyle = "--")
        plt.plot(x_value, results[4][0],label = "$\\epsilon = 0.1$, $\\alpha = 0.5$" + ", best values = " + str(round(results[4][0][-1], 2)), linestyle = "--")
        plt.plot(x_value, results[5][0],label = "$\\epsilon = 0.1$, $\\alpha = 0.9$" + ", best values = " + str(round(results[5][0][-1], 2)), linestyle = "--")
        plt.hlines(true_best, xmin=0, xmax=self.epoch, color="r", label="True best, " + str(true_best))
        plt.title("SARSA-Learning Analysis Plot")   
        plt.ylabel("Tour Distance")
        plt.xlabel("Number of Epocs")
        plt.legend()
        plt.show()
        plt.savefig(f"./figures/Distance_evolution_{tag}")

    def Genetic_agorithm(self):
        
        results= []
        for i in self.mutation_rate: # [0.1, 0.3, 0.7]
            for j in self.population_size: # [10, 30]
                results.append(Genetic_Analysis(epoch=self.epoch, mutation_rate=i, population_size=j))
       
        x_value = [_ for _ in range(self.epoch)]
        plt.figure(figsize=(19, 7))
        plt.plot(x_value, results[0],label = "$mutation rate = 0.1$, $ population size = 10$" + ", best values = " + str(round(results[0][-1], 2)),linestyle = "-")
        plt.plot(x_value, results[1],label = "$mutation rate = 0.1$, $ population size = 30$" + ", best values = " + str(round(results[1][-1], 2)),linestyle = "-")
        plt.plot(x_value, results[2],label = "$mutation rate = 0.3$, $ population size = 10$" + ", best values = " + str(round(results[2][-1], 2)),linestyle = "-")
        plt.plot(x_value, results[3],label = "$mutation rate = 0.3$, $ population size = 30$" + ", best values = " + str(round(results[3][-1], 2)),linestyle = "-")
        plt.plot(x_value, results[4],label = "$mutation rate = 0.7$, $ population size = 10$" + ", best values = " + str(round(results[4][-1], 2)),linestyle = "-")
        plt.plot(x_value, results[5],label = "$mutation rate = 0.7$, $ population size = 30$" + ", best values = " + str(round(results[5][-1], 2)),linestyle = "-")
        # plt.hlines(true_best, xmin=0, xmax=self.epoch, color="r", label="True best, " + str(true_best))
        plt.title("Genetic Algorithm Analysis Plot")   
        plt.ylabel("Tour Distance")
        plt.xlabel("Number of Epocs")
        plt.legend()
        plt.show()

    def ACO(self):
        
        results= []
        for i in self.alpha: # [1, 3]
            for j in self.beta: # [10, 7]
                results.append(ACO_Analysis(epoch=200, alpha=i, beta=j))

        val = len(results[0])
        for i in range(len(results)):
            temp = results[i] + [results[i][-1]] * (self.epoch - val)
            results[i] = temp

        x_value = [_ for _ in range(self.epoch)]
        plt.figure(figsize=(19, 7))
        plt.plot(x_value, results[0],label = "$alpha = 1$, $ beta = 10$",linestyle = "-")
        plt.plot(x_value, results[1],label = "$alpha = 1$, $ beta = 7$",linestyle = "-")
        plt.plot(x_value, results[2],label = "$alpha = 3$, $ beta = 10$",linestyle = "-")
        plt.plot(x_value, results[3],label = "$alpha = 3$, $ beta = 7$",linestyle = "-")
        # plt.hlines(true_best, xmin=0, xmax=self.epoch, color="r", label="True best, " + str(true_best))
        plt.title("Ant Colony Optimisation Analysis Plot")  
        plt.ylabel("Tour Distance")
        plt.xlabel("Number of Epocs")
        plt.legend()
        plt.show()

    def Complete_analysis(self):
        
        result_1 = ACO_Analysis(epoch=200, alpha=1, beta=10)  # best observed value in our analysis
        print("ACO Done ")
        best_aco = result_1[-1]
        val = len(result_1) 
        result_1 = result_1 + [result_1[-1]] * (self.epoch - val)

        result_2 = Genetic_Analysis(epoch=self.epoch, mutation_rate=0.7, population_size=30)
        best_gn = result_2[-1]
        print("Genetic Done")

        result_3,true_best,_ = RL_Methods_Analysis(epsilon=0.1, gamma=0.9, learning_rate=0.5,epochs=self.epoch, method="Q")
        best_q = result_3[-1]
        print("Q Done")

        result_4,_,_ = RL_Methods_Analysis(epsilon=0.1, gamma=0.9, learning_rate=0.5,epochs=self.epoch, method="SARSA")
        best_sar = result_4[-1]
        print("SARSA Done")
    
        x_value = [_ for _ in range(self.epoch)]
        plt.figure(figsize=(19, 7))
        plt.plot(x_value, result_1 ,label = "Ant Colony optimization, $\\alpha = 3$, $ \\beta = 7$" + " best value = " + str(round(best_aco,2)),linestyle = "-")
        plt.plot(x_value, result_2 ,label = "Genetic Algorithm, $mutationrate = 0.7$, $ populationsize = 30$" + " best value = " + str(round(best_gn,2)),linestyle = "-")
        plt.plot(x_value, result_3,label = "Q-learning, $\\epsilon = 0.1$, $\\alpha = 0.5$" + ", best values = " + str(round(best_q, 2)), linestyle = "-")
        plt.plot(x_value, result_4,label = "SARSA-learning, $\\epsilon = 0.1$, $\\alpha = 0.5$" + ", best values = " + str(round(best_sar, 2)), linestyle = "-")
        plt.hlines(true_best, xmin=0, xmax=self.epoch, color="r", label="True best = " + str(true_best))
        plt.title("Comparative Analysis of ACO, Genetic, Q-learning & SARSA learning Algorithm")   
        plt.xlabel("Number of Epocs")
        plt.legend()
        plt.show()

AI_Project = AnalysisPlot()
# RLproject.QL_Reward_Ep()
AI_Project.Q_learning()
# AI_Project.SARSA_learning()
# AI_Project.Genetic_agorithm()
# AI_Project.ACO()
# AI_Project.Complete_analysis()
