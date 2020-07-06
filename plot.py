import matplotlib.pyplot as plt
import torch as T

actor_loss = []
critic_loss = []

path = "/home/Mao/workspace/ATP.ai/06-13:49:17/a-c_150.pth"
trained = T.load(path, map_location='cpu')
# total_scores = trained['score']
total_actor_losses = trained['actor_loss']
total_critic_losses = trained['critic_loss']


# plt.plot(total_scores, label='total_scores')
# plt.plot(test_losses_mean, label='test_loss_mean')
plt.plot(total_critic_losses, label='total_critic_losses')
# plt.plot(test_stress_losses_mean, label='test_stress_losses_mean')
plt.plot(total_actor_losses, label='total_actor_losses')
plt.legend((
            'total_critic_losses',
            'total_actor_losses'
            ),
          loc='upper right', shadow=True)
# plt.yscale("log")
plt.show()