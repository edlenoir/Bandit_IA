import random
from matplotlib import pyplot as plt
from bandit import MovieBandit, SimpleBandit
from agents import SoftmaxAgent, randomAgent, epsGreedyAgent, SoftmaxMulti
def plot_softmax_simple():
    b = SimpleBandit()
    
    temp = 0.1
    for i in range(0, 5) :
        tau = b.run(SoftmaxAgent(temp), 1000)
        plt.plot(range(tau[0]), tau[3], label = "tau "+"{:.2f}".format(temp))
        temp+=0.4
    plt.legend()
    plt.title("Simple Bandit")
    plt.show()
    
    
def plot_softmax_movie():
    b = MovieBandit()

    temp = 0.1
    for i in range(0, 5) :
        b = MovieBandit()
        tau = b.run(SoftmaxMulti(temp))
        plt.plot(range(tau[0]), tau[3], label = "tau "+"{:.2f}".format(temp))
        temp+=0.4
        print(tau[3])
        print("score total : ", tau[1])
    plt.legend()
    plt.title("Movie Bandit")
    plt.show()
    
    
def plot_compare_simple() :
    
    b = SimpleBandit()
    r = randomAgent()
    cnt, total_score, total_regret, score_cumul = b.run(r, 1000)
    plt.plot(range(cnt), score_cumul, label = "random")
    
    r2 = epsGreedyAgent(0.1)
    cnt2, total_score2, total_regret2, score_cumul2 = b.run(r2, 1000)
    plt.plot(range(cnt2), score_cumul2, label = "epsGreedy")
    
    r3 = SoftmaxAgent()
    cnt3, total_score3, total_regret3, score_cumul3 = b.run(r3, 1000)
    plt.plot(range(cnt3), score_cumul3, label = "SoftMax")
    
    plt.legend()
    plt.title("Simple Bandit Comparaison")
    plt.show()
    
    
def plot_compare_movie() :
    
    b = MovieBandit()
    r = randomAgent()
    cnt, total_score, total_regret, score_cumul = b.run(r)
    plt.plot(range(cnt), score_cumul, label = "random")
    
    b = MovieBandit()
    r2 = epsGreedyAgent(0.1)
    cnt2, total_score2, total_regret2, score_cumul2 = b.run(r2)
    plt.plot(range(cnt2), score_cumul2, label = "epsGreedy")
    
    b = MovieBandit()
    r3 = SoftmaxMulti()
    cnt3, total_score3, total_regret3, score_cumul3 = b.run(r3)
    plt.plot(range(cnt3), score_cumul3, label = "SoftMax")
    
    plt.legend()
    plt.title("Movie Bandit Comparaison")
    plt.show()
    
    
    
    
    



if __name__ == '__main__':
    #plot_softmax_simple()
    #plot_softmax_movie()
    #plot_compare_simple()
    plot_compare_movie()
    '''
  b = SimpleBandit()
  r = randomAgent()
  cnt, total_score, total_regret = b.run(r, 1000)
  print("Random: total score against simple bandit:", total_score,
        "regret per turn", total_regret / cnt)

  b = SimpleBandit()
  r = epsGreedyAgent(0.1)
  cnt, total_score, total_regret = b.run(r, 1000)
  print("Epsilon-greedy: total score against simple bandit:", total_score,
        "regret per turn", total_regret / cnt)

  b = SimpleBandit()
  r = SoftmaxAgent()
  cnt, total_score, total_regret = b.run(r, 1000)
  print("SoftmaxAgent : total score against simple bandit:", total_score,
        "||regret per turn", total_regret / cnt)

  b = SimpleBandit()
  r = EpsilonDecreasingAgent(start_epsilon=1.0, epsilon_decay=0.995, minimum_epsilon=0.01)
  cnt, total_score, total_regret = b.run(r, 1000)
  print("EpsilonDecreasing: total score against simple bandit:", total_score,
        "regret per turn", total_regret / cnt)

  b = MovieBandit()
  r = randomAgent()
  cnt, total_score, total_regret = b.run(r)
  print("Random: total score against movie bandit:", total_score,
        "regret per turn:", total_regret / cnt)
  b = MovieBandit()
  r = epsGreedyAgent(0.1)
  cnt, total_score, total_regret = b.run(r)
  print("Epsilon-greedy: total score against simple bandit:", total_score,
        "regret per turn:", total_regret / cnt)
  
  b = MovieBandit()
  r = SoftmaxMulti()
  cnt, total_score, total_regret = b.run(r)
  print("SoftMaxMulti: total score against movie bandit:", total_score,
        "regret per turn:", total_regret / cnt)
  
  b = MovieBandit()
  r = EpsilonDecreasingAgentMulti()
  cnt, total_score, total_regret = b.run(r)
  print("EpsilonDecreasing: total score against movie bandit:", total_score,
        "regret per turn:", total_regret / cnt)
  '''
  
  
  


