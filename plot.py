# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import torch as T
from  statistics import mean
from collections import defaultdict

actor_loss = []
critic_loss = []
# change the path to the output file to draw the training loss and score diagram
path = "a-c_7900_j242912-3.pth"
trained = T.load(path, map_location='cpu')

total_actor_losses = trained['actor_loss']
total_critic_losses = trained['critic_loss']
total_scores = trained['total_score']

plt.subplot(3, 1, 1)
plt.plot(total_actor_losses, label='total_actor_losses')
plt.ylabel('actor losses')

plt.subplot(3, 1, 2)
plt.plot(total_critic_losses, label='total_critic_losses')
plt.ylabel('critic losses')

plt.subplot(3, 1, 3)
total_average_scores = [mean(total_scores[i-100:i]) for i in range(100, len(total_scores))]
plt.plot(total_average_scores, label='total_scores')
plt.ylabel('average score')

plt.show()

# print(mean(total_scores[-100:]))
#
# a = defaultdict(int)
# for s in total_scores:
#     a[s] += 1
#
# a_sorted = {k:v for k,v in sorted(a.items(), key=lambda i: i[1]) if k>2}
# print(a_sorted.keys())