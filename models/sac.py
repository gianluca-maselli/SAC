import torch
from torch.optim import Adam
import torch.nn.functional as F
from SAC.critic import QNet
from SAC.actor import Policy
from SAC.common import soft_update, hard_update
import numpy as np

# Soft-Actor-Critic
class SAC(object):
    def __init__(self, state_size, action_size, action_space, device, args):
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.device = device
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        
        #critic net, critic target  and optimizer
        self.critic = QNet(input_size=state_size+action_size, hidden_size=args.hidden_size).to(device=self.device)
        self.critic_target = QNet(input_size=state_size+action_size, hidden_size=args.hidden_size).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
        hard_update(self.critic_target, self.critic)
        
        #policy net
        if args.policy == 'Gaussian':
            """
            Automatic entropy tuning avoids to manually tune the entropy according to the task
            and the magnitude of the rewards. As stated in the paper, forcing the entropy
            to a fixed value is a poor solution, since the policy should be free to
            explore more in regions where the optimal action is uncertain, but remain
            more deterministic in states with a clear distinction between good and bad actions.
            """
            if self.automatic_entropy_tuning is True:
                # target entropy is a heuristic value: −dim(A), where A is the action space dimensionality (e.g. , -6 for HalfCheetah-v2) as given in the paper)
                # encourages more exploration in larger action spaces
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                # Initialize log(alpha), the log of the entropy temperature, as a trainable parameter
                # Keeping it in log-space helps with numerical stability and ensures alpha > 0
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                # Optimizer to adjust log_alpha during training
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
            #use of the Gaussian Policy
            self.actor = Policy(input_size=state_size, num_actions=action_size, hidden_size=args.hidden_size, mode=args.policy, action_space=action_space).to(self.device)
        else:
            #setting entropy manually
            self.alpha = 0
            self.automatic_entropy_tuning = False
            #use Deterministic policy
            self.actor = Policy(input_size=state_size, num_actions=action_size, hidden_size=args.hidden_size, mode=args.policy, action_space=action_space).to(self.device)
        
        self.actor_optim = Adam(self.actor.parameters(), lr=args.lr)
    
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.actor.sample(state)
        else:
            _, _, action = self.actor.sample(state)
        
        return action.detach().cpu().numpy()[0]

    def update(self, replay_buffer, updates):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = replay_buffer.sample()
        #computation of the Q-function
        #starting by computing V(s_t+1), i.e. the soft state value function
        # V(s_t+1) = Q(s_t+1, a_t+1) - alpha*log_pi(a_t+1∣s_t+1)
        with torch.no_grad():
            # sample next actions a_t = π(s_t+1) and compute logπ(a_t+1|s_t+1)
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_state_batch)
            # compute the Q_values for the next state
            q1_next_target, q2_next_target = self.critic_target(next_state_batch, next_state_action)
            # taking the minimum of the two Q-values is the Clipped Double Q-learning trick, which helps reduce overestimation bias.
            min_q_next_target = torch.min(q1_next_target, q2_next_target) - self.alpha * next_state_log_pi
            # summing R(s_t, a_t) + y*V(s_t+1) (bootstrapped target)
            next_q_value = reward_batch + mask_batch * self.gamma * (min_q_next_target)
        #compute the overall loss function
        # JQ = 𝔼(st,at)~D[1/2(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        # two Q-functions that minimize the Bellman residual for both. The two network are trained independently
        q1, q2 = self.critic(state_batch, action_batch)
        #compute the loss for each of the critic to allow indipendent training.
        q1_loss = F.mse_loss(q1, next_q_value)
        q2_loss = F.mse_loss(q2, next_q_value)
        q_loss = q1_loss + q2_loss
        #backpropagate gradients for the critics
        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()

        #update actor (policy)
        # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        # sample actions a_t = π(s_t) from the policy using the reparameterization trick.
        # get logπ(a_t|s_t), i.e. the log-probability of the sampled action.
        pi, log_pi, _ = self.actor.sample(state_batch)
        # compute the Q-values (q1 and q2) for the sampled action a_t = π(s_t)
        q1_pi, q2_pi = self.critic(state_batch, pi)
        # use clipped double Q-learning (take min) to reduce overestimation bias.
        min_q_pi = torch.min(q1_pi, q2_pi)
        #compute the final loss: Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        actor_loss = ((self.alpha * log_pi) - min_q_pi).mean()
        #backpropagate gradients for the actor
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        #entropy coefficient tuning
        if self.automatic_entropy_tuning:
            """
            case of automatic entropy coefficient tuning.
            instead of keeping alpha fixed, SAC can learn alpha to match a target entropy.
            this makes the algorithm adaptively explore or exploit depending on how confident the policy is.
            """
            # the objective to minimize is: J(α) = 𝔼at∼πt[-α * logπ(at|st) - αH_target]
            # ideally, logπ(at|st) ≈ H_target so that entropy matches the desired level.
            # if the actual entropy is too low, the loss increases and α goes up (more exploration).
            # if entropy is too high, α goes down (less randomness).
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            #compute gradients to update the entropy coefficient
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            #Convert back from log space to actual α value. This is what gets used in the actor loss and critic target calculation
            self.alpha = self.log_alpha.exp().detach()
            alpha_tlogs = self.alpha.clone()
        else:
            # if automatic tuning is not enabled, use a fixed α value and set alpha_loss to zero(no update)
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)

        #use soft update to sync the target critic with the weights of the critic
        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return q1_loss.item(), q2_loss.item(), actor_loss.item(), alpha_loss.item(), alpha_tlogs.item()

