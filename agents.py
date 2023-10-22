import random
import numpy as np


class randomAgent:

  def present(self, context, choices):
    return random.choice(choices)

  def feedback(self, score):
    pass


class SoftmaxAgent:

  def __init__(self, temperature=1.0):
    self.temperature = temperature
    self.value_estimates = {}
    self.action_counts = {}

  def present(self, context, choices):
    # Si de nouvelles actions sont présentes, initialisez leurs estimations
    for choice in choices:
      if choice not in self.value_estimates:
        self.value_estimates[choice] = 0
        self.action_counts[choice] = 0

    values = np.array([self.value_estimates[choice] for choice in choices])
    probabilities = np.exp(values / self.temperature) / np.sum(
        np.exp(values / self.temperature))
    self.last_action = np.random.choice(choices, p=probabilities)
    return self.last_action

  def feedback(self, score):
    # Mettre à jour l'estimation de la valeur pour l'action choisie
    chosen_action = self.last_action
    self.action_counts[chosen_action] += 1
    alpha = 1.0 / self.action_counts[chosen_action]
    current_estimate = self.value_estimates[chosen_action]
    self.value_estimates[chosen_action] = current_estimate + alpha * (
        score - current_estimate)


class epsGreedyAgent():

  def __init__(self, epsilon):
    self.epsilon = epsilon
    self.last_choice = 0  #just a random initialization
    self.counts = dict()  # Count represents counts of pulls for each arm.
    self.values = dict()  # Value represents average reward for a specific arm.

  def present(self, context, choices):
    # If prob is not in epsilon and we've seen at least some of the options, do exploitation of best arm so far
    if random.random() > self.epsilon and sum([
        self.counts.get(c, 0) for c in choices
    ]) > (len(choices) /
          2):  #avoid exploiting while there is little or no data
      # assume that choices are a subset of available keys
      self.last_choice = max(choices, key=lambda k: self.values.get(k, 0))
    # If prob falls in epsilon range, do exploration
    else:
      self.last_choice = random.choice(choices)
    self.counts[self.last_choice] = self.counts.get(self.last_choice,
                                                    0) + 1  #update counts
    return self.last_choice

  def feedback(self, score):
    # calculate mean score for the chosen arm
    old_value = self.values.get(self.last_choice, 0)
    n = self.counts[self.last_choice]
    new_value = ((n - 1) * old_value + score) / n
    self.values[self.last_choice] = new_value

  def load_datam(self):
    movies = dict()
    with open('movies.csv') as csvfile:
      moviereader = csv.reader(csvfile, delimiter=',')
      cnt = 0
      for row in moviereader:
        mid = row[0]
        genres = row[2].split('|')
        if genres[0] != '(no genres listed)':
          cnt += 1
          movies[mid] = genres
      print(cnt, "movies listed")
    return movies





class SoftmaxMulti:

  def __init__(self, temperature=1.0):
    self.temperature = temperature
    self.value_estimates = {}
    self.action_counts = {}

  def present(self, user, choices):
    for choice in choices:
      if choice not in self.value_estimates:
        self.value_estimates[choice] = 0
        self.action_counts[choice] = 0
    values = np.array([self.value_estimates[choice] for choice in choices])
    probabilities = np.exp(values / self.temperature) / np.sum(
        np.exp(values / self.temperature))
    self.last_choice = np.random.choice(choices, p=probabilities)
    return self.last_choice

  def feedback(self, reward):
    # Mettre à jour l'estimation de la valeur pour l'action choisie
    # Note: vous pourriez vouloir ajouter un taux d'apprentissage pour contrôler la mise à jour.
    chosen_movie = self.last_choice
    n = self.action_counts[chosen_movie]
    self.value_estimates[chosen_movie] += (
        reward - self.value_estimates[chosen_movie]) / (n + 1)
    self.action_counts[chosen_movie] += 1
    
  def load_datam(self):
    movies = dict()
    with open('movies.csv') as csvfile:
      moviereader = csv.reader(csvfile, delimiter=',')
      cnt = 0
      for row in moviereader:
        mid = row[0]
        genres = row[2].split('|')
        if genres[0] != '(no genres listed)':
          cnt += 1
          movies[mid] = genres
      print(cnt, "movies listed")
    return movies
    
