############################################################################
############################################################################
# THIS IS THE ONLY FILE YOU SHOULD EDIT
#
#
# Agent must always have these five functions:
#     __init__(self)
#     has_finished_episode(self)
#     get_next_action(self, state)
#     set_next_state_and_distance(self, next_state, distance_to_goal)
#     get_greedy_action(self, state)
#
#
# You may add any other functions as you wish
############################################################################
############################################################################

import numpy as np
import torch
import collections

#The basic network class
class Network(torch.nn.Module):

    # The class initialisation function. 
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has three hidden layers, each with 50 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=50)
        self.layer_2 = torch.nn.Linear(in_features=50, out_features=50)
        self.layer_3 = torch.nn.Linear(in_features=50, out_features=50)
        self.output_layer = torch.nn.Linear(in_features=50, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. 
    # In this example, a ReLU activation function is used for hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        layer_3_output = torch.nn.functional.relu(self.layer_3(layer_2_output))
        output = self.output_layer(layer_3_output)
        return output
    
class DQN:

    # The class initialisation function.
    def __init__(self, step_length):
        # Create a Q-network and target Q-network. Both have two inputs(state x & y location) and four outputs(four Q values)
        self.q_network = Network(input_dimension=2, output_dimension=4)
        self.q_target_network = Network(input_dimension=2, output_dimension=4)
        
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        
        # Define the length of step the agent take for each action
        self.step_length = step_length
        
    # Function that is called whenever we want to train the Q-network. 
    def train_q_network(self, batch_size, batch_input, batch_actions, batch_rewards, next_state_tensor):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        
        # Calculate the loss for training the network.
        loss = self._calculate_loss(batch_size, batch_input, batch_actions, batch_rewards, next_state_tensor)
        
        # Compute the gradients based on this loss.
        loss.backward()
        
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        
        # Get the new weights for this batched transitions in replay buffer, after the Q-network has been trained
        losses_for_buffer = self._calculate_new_weights(batch_input, batch_actions, batch_rewards, next_state_tensor)
        
        # Return new losses for each state in the buffer, for updating each's weight in the replay buffer

        return losses_for_buffer

    # Function to calculate the loss for a batch.
    def _calculate_loss(self, batch_size, batch_input, batch_actions, batch_rewards, next_state_tensor):
        # Get the Q values for each state in this batch
        q_val = self.q_network.forward(batch_input)
       
        # Get the Q values for post states
        q_val_next = self.q_network.forward(next_state_tensor)
        
        # Get the Q values from the target Q-network
        q_val_next_target = self.q_target_network.forward(next_state_tensor)
        
        # Use .detach() to prevent the impact of backward propogation of the Q-network
        q_val_next_target = q_val_next_target.detach()
        
        # Choose the action with highest Q value in the target network
        q_val_action_max = torch.argmax(q_val_next_target, dim=1)
        
        # Get corresponding Q values in the regular network
        q_val_next_max = q_val_next.gather(dim=1, index=q_val_action_max.unsqueeze(-1)).squeeze(-1)
        
        # Get Q(S,A)
        q_val_reward = q_val.gather(dim=1, index=batch_actions.unsqueeze(-1)).squeeze(-1)
        
        # Get (R+Q(S'+ gamma*argmaxQ'(S',a))). Here gamma is 0.95.
        future_rewards = batch_rewards + (0.95 * q_val_next_max)
        
        # Calculate the loss. Here the loss is RMS error.
        loss = torch.nn.MSELoss()(future_rewards, q_val_reward)
        
        # Calculate the loss for each state for updating the weights in replaybuffer
        return loss
    
    def epsilon_greedy(self, state, num_episodes=0, const=0):
        # set the step length of the agent
        step_length = self.step_length
        
        # Get the Q values for this state
        q_val = self.q_network.forward(torch.unsqueeze(torch.tensor(state), 0))
        
        # The greedy action
        greedy_action = int(torch.argmax(q_val))
        
        if(num_episodes == 0 or const == 0):
            # If num_episode is set to zero, or the epsilon constant becomes zero, the action is choosen fully greedy
            epsilon = 0
        else:
            # If the network is still being trained, epsilon is decaying with the increasing number of episodes
            # The epsilon constant decreases every time the agent reaches goal
            epsilon = 1 / (num_episodes) + const
            if epsilon > 1:
                epsilon = 1
        
        # Get the propobility of choosing each action
        next_action_prob = np.zeros(4)
        
        # Larger probabilities are assigned to up and down movements. The agent will not go left, unless it is the greedy action.
        next_action_prob[0] += 4 * epsilon / 9
        next_action_prob[1] += epsilon / 9
        next_action_prob[2] += 4 * epsilon / 9
        next_action_prob[greedy_action] += (1 - epsilon)
        
        # Get the chosen discrete action and turn it to the continuous action
        discrete_action = np.random.choice([0, 1, 2, 3], size=1, replace=False, p=next_action_prob)   
        if discrete_action == 0:
            continuous_action = np.array([0, step_length], dtype=np.float32)
        elif discrete_action == 1:
            continuous_action = np.array([step_length, 0], dtype=np.float32)
        elif discrete_action == 2:
            continuous_action = np.array([0, 0-step_length], dtype=np.float32)
        elif discrete_action == 3:
            continuous_action = np.array([0-step_length, 0], dtype=np.float32)
        else:
            print('Wrong action chosen!')
            return
        
        # Return the continuous action to get next state.
        return continuous_action

    # Calculate the loss for one state. This loss is then added into the replay buffer
    def buffer_loss(self, state, next_state, action, reward):
        # Get Q values for this state
        q_val = self.q_network.forward(torch.unsqueeze(torch.tensor(state), 0))
        
        # Get Q values for the post state
        q_val_next = self.q_network.forward(torch.unsqueeze(torch.tensor(next_state), 0)).detach().numpy()
        
        # Get the Q value for action A
        q_reward = q_val.gather(dim=1, index=torch.unsqueeze(torch.tensor([action], dtype=torch.int64), 0)).squeeze(1)
        
        # Turn this value into float number
        q_reward = float(q_reward)
        
        # The actual reward plus post state Q values
        reward_np = np.zeros(4) + reward
        future_reward = reward_np + 0.95 * q_val_next
        
        # The weight is the maximum loss
        loss_for_buffer = np.max(abs(future_reward - q_reward)+0.00001)
        
        return loss_for_buffer

    # calculate new weights for updating the replay buffer
    def _calculate_new_weights(self, batch_input, batch_actions, batch_rewards, next_state_tensor):
        # Get Q(S,A)
        q_val = self.q_network.forward(batch_input)
        q_val_reward = q_val.gather(dim=1, index=batch_actions.unsqueeze(-1)).squeeze(-1).detach().numpy()
        
        # Get the values for post states
        q_val_next = self.q_network.forward(next_state_tensor).detach().numpy()
        
        # Turn the actural rewards tensor to a numpy variable 
        rewards = batch_rewards.detach().numpy()
        
        # Get the shape of Q values of post states
        shape = q_val_next.shape
        
        # Get the loss numpy theta, for containing all losses of four actions for one state
        theta = np.zeros(shape)
        for i in range(int(shape[1])):
            theta[:, i] = abs(rewards + 0.95 * q_val_next[:, i] - q_val_reward) + 0.00001
        
        # Choose the maximum values from theta, and take it as the new weights
        losses_for_buffer = np.max(theta, axis=1)
        
        return losses_for_buffer
        
    # Function for udating the target network    
    def update_target_network(self):
        torch.save(self.q_network.state_dict(), 'Q_network_parameters.pkl')
        self.q_target_network.load_state_dict(torch.load('Q_network_parameters.pkl'))
    
# The replay buffer class
class ReplayBuffer:
    # The buffer capacity is 10,000 (i.e. able to contain 10,000 transitions)
    buffer = collections.deque(maxlen=10000)
    weight = collections.deque(maxlen=10000)
    
    # The initialization function
    def __init__(self):
        self.buffer_size = 0
    
    # Function for adding a transition and its weight in replay buffer
    def add_transition(self, transition, loss_for_buffer):
        self.buffer.append(transition)
        self.weight.append(loss_for_buffer)

    # Sampling from the replay buffer
    def buffer_sample(self, num_steps):
        # Before sampling, get the propobility of sampling based on the weights of transitions
        p = self.weight.copy()
        weight_sum = sum(self.weight)
        for idx in range(len(self.weight)):
            p[idx] = self.weight[idx] / weight_sum        
        
        # The mini batch size is incresing with the increasing of buffer size
        batch_size = int(len(self.buffer)/20)+1
        # But when maximum mini batch size is 128
        if(batch_size > 128):
            batch_size = 128
        
        # Get sampled transitions
        minibatch_indices = np.random.choice(range(len(self.buffer)), batch_size, False, p)
        
        # Extract data from sampled transitions, and convert them into torch tensors
        minibatch_inputs = [self.buffer[idx][0] for idx in minibatch_indices]
        actions = [self.buffer[idx][1] for idx in minibatch_indices]
        rewards = [self.buffer[idx][2] for idx in minibatch_indices]
        minibatch_input_tensor = torch.tensor(minibatch_inputs)
        action_tensor = torch.tensor(actions, dtype=torch.int64)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        next_state_tensor = torch.tensor([self.buffer[idx][3] for idx in minibatch_indices])
        
        # Return data of sampled transitions
        return batch_size, minibatch_indices, minibatch_input_tensor, action_tensor, reward_tensor, next_state_tensor
    
    # Function for updating the weight of each transition after training the network        
    def update_weight(self, losses_for_buffer, indices):
        i = 0
        for idx in indices:
            self.weight[idx] = losses_for_buffer[i]
            i += 1

# The agent is defined in this class
class Agent:
    # Function to initialise the agent
    def __init__(self):
        # Set the episode length
        self.episode_length = 640
        
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None
        
        # The length of the agent's each step
        self.step_length = 0.02
        
        # The neural network used to train the agent
        self.dqn = DQN(self.step_length)
        
        # The replay buffer for training
        self.replay_buffer = ReplayBuffer()
        
        # The target network is updated every ten episodes
        self.network_update_size = 10
        
        # The early stop flag. Once the training is stopped, it will not start again.
        self.early_stop = False
        
        # The flag for early testing. When it is ture, the current Q-network will be tested.
        self.early_test = False
        
        # Flag to justify if it is the first time the agent reaches the goal in one episode
        self.first_time_in_episode = True
        
        # The epsilon constant to determine the low limit of epsilon in e-greedy function.
        self.epsilon_const = 0.54
        
        # The variable for counting the number of episodes taken
        self.episode_num = 0
        
    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if self.num_steps_taken % self.episode_length == 0:
            self.first_time_in_episode = True
            return True
        else:
            return False

    # Function to get the next action.
    def get_next_action(self, state):
        # The action is determined by epsilon-greedy policy
        action = self.dqn.epsilon_greedy(state, self.episode_num+1, self.epsilon_const)
        
        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
        
        # Store the state; this will be used later, when storing the transition
        self.state = state
        
        # Store the action; this will be used later, when storing the transition
        self.action = action
        
        return action

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        # If the agent reached the goal in a episode, the length of episode and epsilon constant will be decreased.
        if(distance_to_goal < 0.03 and self.first_time_in_episode and (not self.early_test)):
            self.epsilon_const -= 0.01
            self.episode_length -= 10
            # However, they will not be less than 0 and 100, respectively.
            if self.epsilon_const < 0:
                self.epsilon_const = 0
            if self.episode_length < 100:
                self.episode_length = 100
            # Once the agent reaches the goal, this flag will become false, 
            # so that the episode length and epsilon constant will not be reduced more than once in the same episode.    
            self.first_time_in_episode = False
        
        # The test will be done every five episodes, only after episode length == 100, epsilon constant == 0(fully greedy), and the training has yet been stopped.
        if((self.episode_num+1) % 5 == 0 and self.episode_length == 100 and self.epsilon_const == 0 and self.first_time_in_episode and (not(self.early_stop))):
            # If this flag = True, it will pause the training and use current network to do the test.
            self.early_test = True
        else:
            self.early_test = False
        
        # The reward funtion determined by distance to goal
        reward = float(0.5*(1 - distance_to_goal**0.5))
        
        # Before creating a transition, convert the continouous action to discrete action
        # Which will be used when calculating losses
        discrete_action = self.get_discrete_actions(self.action)
        
        # Create the transition
        transition = (self.state, discrete_action, reward, next_state)
        
        # If the programme is in agent test mode (in this mode the episode length == 100 and epsilon == 0), 
        # when the agent reached the goal, the training will be stopped until the end
        # This is to prevent the instablized network due to over-training.
        if (self.early_test):
            if distance_to_goal < 0.03:
                self.early_stop = True
                print('The training has been early finished because the agent already knows how to reach the goal. It will not be started again during this execution.')
        
        # Only when the training is not stopped and the agent is not in test mode, the following fuctions will be excuted.
        if not (self.early_stop or self.early_test):
            # Get the weight of this new state
            loss_for_buffer = self.dqn.buffer_loss(self.state, next_state, discrete_action, reward)
            
            # Add the transition and new wight into replay buffer
            self.replay_buffer.add_transition(transition, loss_for_buffer)
            
            # Sampling from replay buffer
            batch_size, indices, batch_input, batch_actions, batch_rewards, next_state_tensor = self.replay_buffer.buffer_sample(self.num_steps_taken)
            
            # Use sampled batch to train the network
            losses_for_buffer = self.dqn.train_q_network(batch_size, batch_input, batch_actions, batch_rewards, next_state_tensor)
            
            # Update each transition's weight in replay buffer.
            self.replay_buffer.update_weight(losses_for_buffer, indices)
            
            # Update the target network every 10 episodes
            if(self.num_steps_taken % self.episode_length == 0):
                self.dqn.update_target_network()                  
        
        # Update the count for the number of episodes
        if(self.num_steps_taken % self.episode_length == 0):
            self.episode_num += 1

    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        # Here, the action will be fully greedily (epsilon = 0) selected, since the number of episode is set to 0.
        action = self.dqn.epsilon_greedy(state, 0)
        return action
    
    # This function convert the continuous action into discrete action
    # The discrete action is added into the transition for batch sampling and calculating losses
    def get_discrete_actions(self, action):
        if (action[0] == 0 and action[1] > 0):
            discrete_action = 0
        elif(action[0] > 0 and action[1] == 0):
            discrete_action = 1
        elif(action[0] == 0 and action[1] < 0):
            discrete_action = 2
        elif(action[0] < 0 and action[1] == 0):
            discrete_action = 3
        else:
            print('Something wrong')
        return discrete_action
###############################################################################################
        
    def is_early_stop(self):
        if self.early_stop:
            return True
        else:
            return False