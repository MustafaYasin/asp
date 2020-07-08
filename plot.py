import matplotlib.pyplot as plt
import torch as T

actor_loss = []
critic_loss = []
# change the path to the output file to draw the training loss and score diagram
path = "0707_21:36/a-c_50.pth"
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
plt.plot(total_scores, label='total_scores')
plt.ylabel('score')

plt.show()