import csv
import random
import math
from collections.abc import Iterator, Generator


class SimpleBandit:

  def __init__(self):
    self.outcomes = {
        'A': {
            1: 0.7,
            4: 0.1,
            0: 0.2
        },
        'B': {
            6: 0.25,
            3: 0.25,
            0: 0.5
        },
        'C': {
            9: 0.3,
            3: 0.1,
            1: 0.15,
            0: 0.65
        },
        'D': {
            11: 0.04,
            2: 0.16,
            1: 0.15,
            0: 0.65
        }
    }

  def run(self, agent, n=100):
    cnt = 0
    total_score = 0
    score_cumul = []
    choices = list(self.outcomes.keys())
    expectations = [
        sum([k * self.outcomes[choice][k] for k in self.outcomes[choice]])
        for choice in self.outcomes
    ]
    #print (expectations)
    best_expectation = max(expectations)
    for i in range(n):
      cnt += 1
      choice = agent.present(0, choices)  #context is not used
      possibleScores = list(self.outcomes[choice].keys())
      probs = [self.outcomes[choice][k] for k in possibleScores]
      score = random.choices(possibleScores, weights=probs)[0]
      agent.feedback(score)
      total_score += score
      score_cumul.append(total_score)
    total_regret = cnt * (best_expectation) - total_score

    return cnt, total_score, total_regret, score_cumul


class MovieBandit(Generator):

  def __init__(self):
    self.generator_iterator = self.generator()

  def send(self, ignored_argument):
    return next(self.generator_iterator)

  def generator(self):
    with open('training2.txt') as csvfile:
      moviereader = csv.reader(csvfile, delimiter=',')
      cnt = 0
      experiments = []
      for row in moviereader:
        uid = row[0]
        choices = row[1].split('|')
        values = [float(v) for v in row[2].split("|")]
        yield uid, {c: float(v) for c, v in zip(choices, values)}

  def throw(self, type=None, value=None, traceback=None):
    raise StopIteration

  def run(self, agent):
    cnt = 0
    total_score = 0
    total_regret = 0
    score_cumul = []
    for u, ratings in self.generator_iterator:
      cnt += 1
      choices = list(ratings.keys())
      choice = agent.present(u, choices)
      score = ratings[choice]
      best = max([ratings[c] for c in choices])
      agent.feedback(score)
      total_score += score
      score_cumul.append(total_score)
      total_regret += best - score

    return cnt, total_score, total_regret, score_cumul
