import numpy as np;
import matplotlib.pyplot as plt;
import random;

class GridWorld(object):
    def __init__(self):
#the properties of this grid world
        self.shape = (6,6)
        self.obstacle_locs = [(1,1), (2,3), (2,5), (3,1), (4,1), (4,2), (4,4)]
        self.terminal_locs = [(1,3), (4,3)]
        self.special_rewards = [10, -100]
        self.default_rewards = -1
        self.action_names = ['N', 'E', 'S', 'W']
        self.action_size = len(self.action_names)
        self.prop_desire = 0.5
        self.prop_oth = (1 - self.prop_desire)/3
        self.discount = 0.2
        self.threshold = 0.001
        self.state_size, self.tran_matrix, self.reward_matrix, self.terminal, self.neighbours = self.build()
#check whether this location is a state or a obstacle
    def is_location(self, loc):
        if (loc[0]<0 or loc[1]<0 or loc[0]>self.shape[0]-1 or loc[1]>self.shape[1]-1):
            return False
        elif (loc in self.obstacle_locs):
            return False
        else:
            return True
#convert a location to a state number
    def loc_to_state(self, loc, locs):
        return locs.index(tuple(loc))
#get the neighbour states of a state
    def get_neighbours(self, loc, direction):
        i = loc[0]
        j = loc[1]
        n = (i-1, j)
        e = (i, j+1)
        s = (i+1, j)
        w = (i, j-1)
        if(direction == 'n' and self.is_location(n)):
            return n
        elif(direction == 'e' and self.is_location(e)):
            return e
        elif(direction == 's' and self.is_location(s)):
            return s
        elif(direction == 'w' and self.is_location(w)):
            return w
        else:
            return loc
#get the locations of each state. Also the neighbour state matrix and the location of terminal states
    def get_locations(self):
        height = self.shape[0]
        width = self.shape[1]
        locs = []
        neighbour_locs = []
        for i in range(height):
            for j in range (width):
                loc = (i, j)
                if (self.is_location(loc)):
                    locs.append(loc)
                    local_neighbours = [self.get_neighbours(loc, direction) for direction in ['n', 'e', 's', 'w']]
                    neighbour_locs.append(local_neighbours)
        num_states = len(locs)
        state_neighbours = np.zeros((num_states, 4))
        for state in range (num_states):
            for direction in range (4):
                nloc = neighbour_locs[state][direction]
                nstate = self.loc_to_state(nloc, locs)
                state_neighbours[state, direction] = nstate
        terminal = np.zeros((1,num_states))
        for a in self.terminal_locs:
            terminal_state = self.loc_to_state(a, locs)
            terminal[0, terminal_state] = 1
        return locs, state_neighbours, terminal
#Build the grid world with above properties. Return reward matrix and transition matrix
    def build(self):
        locations, neighbours, terminal = self.get_locations()
        S = len(locations)
        tran_matrix = np.zeros((S, S, 4))
        for action in range (4):
            for effect in range(4):
                for state in range(S):
                    post_state = neighbours[state, effect]
                    post_state = int(post_state)
                    if(action == effect):
                        tran_matrix[state, post_state, action] = tran_matrix[state, post_state, action] + self.prop_desire
                    else:
                        tran_matrix[state, post_state, action] = tran_matrix[state, post_state, action] + self.prop_oth
        reward_matrix = self.default_rewards * np.ones((S, S, 4))       
        for i, r in enumerate (self.special_rewards):
            reward_state = self.loc_to_state(self.terminal_locs[i], locations)
            reward_matrix[:, reward_state, :] = r      
        return S, tran_matrix, reward_matrix, terminal, neighbours
#Using Dynamic Programming to evaluate a policy
    def policy_evaluation_dp (self, policy):
        delta = 2 * self.threshold
        T = self.tran_matrix
        R = self.reward_matrix
        V = np.zeros(self.state_size)
        Vnew = np.copy(V)
        while delta>self.threshold:
            for state_idx in range(self.state_size):
                if(self.terminal[0, state_idx]):
                    continue
                temV = 0
                for action_idx in range(policy.shape[1]):
                    temQ = 0
                    for state_idx_post in range (policy.shape[0]):
                        temQ += (T[state_idx, state_idx_post, action_idx] * (R[state_idx, state_idx_post, action_idx] + self.discount * V[state_idx_post]))
                    temV += (policy[state_idx, action_idx] * temQ)
                Vnew[state_idx] = temV
            delta = max(abs(Vnew - V))            
            V = np.copy(Vnew)        
        return V
#Improve the current policy towards the optimal policy
    def policy_improvement_dp (self, value):
        T = self.tran_matrix
        R = self.reward_matrix
        policy = np.zeros((self.state_size, self.action_size))
        for state_idx in range(self.state_size):
            if not (self.terminal[0,state_idx]):
                Q = np.zeros(4)
                for state_idx_post in range(self.state_size):
                    Q += (T[state_idx, state_idx_post, :] * (R[state_idx, state_idx_post, :] + self.discount * value[state_idx_post]))
                new_policy = np.zeros(4)
                new_policy[np.argmax(Q)] = 1
                policy[state_idx] = new_policy
        return policy
#Using DP to get the optimal value functions    
    def get_true_value(self):
        policy = 0.25 * np.ones((self.state_size, self.action_size))
        policy_state = True
        while policy_state:
            temPolicy = np.copy(policy)
            Value = self.policy_evaluation_dp(policy)
            policy = self.policy_improvement_dp(Value)
            if((temPolicy == policy).all()):
                policy_state = False
            else:
                policy_state = True
        return Value
#Get an episode with a starting state
    def get_episode(self, starting_states, policy):
        episode = []
        dup_state_in_episode=[]
        state = random.choice(starting_states)
        while(True):
            action, post_state, reward = self.next_state_mc(state, policy, self.neighbours)
            episode.append((int(state), action, reward))
            dup_state_in_episode.append(state)
            if (reward == 10 or reward == -100):
                break
            state = post_state
        return episode
#For MC, get the post state number, action, and reward. Called in the get_episode function
    def next_state_mc(self, state, policy, neighbours):
        policy_p = np.copy(policy[state, :])
        action = np.random.choice([0, 1, 2, 3], size=1, replace=False, p=policy_p.ravel())
        action = int(action)
        action_p = [0, 0, 0, 0]
        for i_action in range(4):
            if(i_action == action):
                action_p[i_action] = self.prop_desire
            else:
                action_p[i_action] = self.prop_oth
        movement = np.random.choice([0, 1, 2, 3], size=1, replace=False, p=action_p)
        movement = int(movement)
        post_state = neighbours[state,movement]
        post_state = int(post_state)
        if(post_state == 8):
            reward = 10
        elif(post_state == 21):
            reward = -100
        else:
            reward = -1
        return action, post_state, reward
#Using MC to evaluate the policy
    def policy_evaluation_mc(self, policy):
        starting_states = list(range(self.state_size))
        terminal_states = [8, 21]
        for del_state in terminal_states:
            starting_states.remove(del_state)        
        num_episode = 2000
        V = np.zeros(self.state_size)
        state_sum = np.zeros(self.state_size)
        return_sum = np.zeros(self.state_size)
        for i_episode in range(1, num_episode+1):
            episode = []
            dup_state_in_episode=[]
            state = random.choice(starting_states)
            for t in range (100):
                action, post_state, reward = self.next_state_mc(state, policy, self.neighbours)
                episode.append((int(state), action, reward))
                dup_state_in_episode.append(state)
                if (reward == 10 or reward == -100):
                    break
                state = post_state
            state_in_episode = set(dup_state_in_episode)
            for i_state in state_in_episode:
                first_idx = next(i for i,x in enumerate(episode) if x[0] == i_state)
                first_sum = sum(x[2]*(self.discount**i) for i,x in enumerate(episode[first_idx:]))
                state_sum[i_state] += first_sum
                return_sum[i_state] += 1.0
                V[i_state] = state_sum[i_state] / return_sum[i_state]
        return V
#epsilon-greedy algorithm for improving the policy
    def policy_improvement(self, Q):
        epsilon = 0.05
        policy = np.zeros((self.state_size, self.action_size))+(epsilon / self.action_size)
        temQ = np.zeros(self.action_size)
        for state_idx in range(self.state_size):
            if not (self.terminal[0,state_idx]):
                temQ = Q[state_idx, :]
                policy[state_idx, np.argmax(temQ)] += (1-epsilon)
        return policy
#MC method to estimate the optimal solution. Note that here the returns are for poltting the learning curve.
    def policy_iteration_mc(self, num_episode, step):
        starting_states = list(range(self.state_size))
        terminal_states = [8, 21]
        for del_state in terminal_states:
            starting_states.remove(del_state)       
        policy = 0.25 * np.ones((grid.state_size, grid.action_size))
        # policy[:, 2] = 1
        Q = np.zeros((self.state_size, self.action_size))
        alpha = 0.05
        episode_return = []
        mean = []
        stdv = []        
        for i_episode in range(1, num_episode+1):
            episode = self.get_episode(starting_states, policy)
            G = 0            
            for i_state in range(len(episode)):
                state, action, reward = episode [i_state]
                G = self.discount * G + reward
                Q[state, action] += alpha * (G - Q[state, action])                   
            episode_return.append(G)
            mean.append(G)
            stdv.append(np.std(episode_return))                
            policy = self.policy_improvement(Q)            
        mean_np = np.array(mean)
        std_np = np.array(stdv)
        mean_plus_stdv = mean_np + std_np
        mean_minus_stdv = mean_np - std_np
        return mean_np, mean_plus_stdv, mean_minus_stdv
#Drawing the learning curve for MC method    
    def learning_curve_mc (self, num_episode, step, num_curve):
        mean_sum = np.zeros((num_curve, int(num_episode/step)))
        mean_plus_sum = np.copy(mean_sum)
        mean_minus_sum = np.copy(mean_sum)
        for i_curve in range (num_curve):
            mean, mean_plus_stdv, mean_minus_stdv = self.policy_iteration_mc(num_episode, step)
            mean_sum[i_curve, :] = mean
            mean_plus_sum[i_curve, :] = mean_plus_stdv
            mean_minus_sum[i_curve, :] = mean_minus_stdv
            if((i_curve+1) % 10 == 0):
                print("learning curves:{}".format(i_curve+1))
        mean_plot = sum(mean_sum)/num_curve
        mean_plus_plot = sum(mean_plus_sum)/num_curve
        mean_minus_plot = sum(mean_minus_sum)/num_curve       
        x_axis = np.arange(step, num_episode+step, step)
        plt.figure(figsize=(10,6))
        plt.subplot(1, 2, 1)
        plt.plot(x_axis, mean_plot, label='Mean')
        plt.legend(loc = 'lower right')
        plt.xlabel("Number of Episodes")
        plt.ylabel("Mean Reward")
        plt.title("Learning curve for a=0.05 & e=0.05")
        plt.grid(linestyle = ':')
        plt.subplot(1, 2, 2)
        plt.plot(x_axis, mean_plot, color='r', linestyle='-', label='Mean')
        plt.plot(x_axis, mean_plus_plot, color='b', linestyle=':', label='Mean+Stdv')
        plt.plot(x_axis, mean_minus_plot, color='g', linestyle=':', label='Mean-Stdv')      
        plt.legend(loc = 'lower right')
        plt.xlabel("Number of Episodes")
        plt.ylabel("Mean Reward & Mean Reward+/-Stdv")
        plt.grid(linestyle = ':')
        plt.show()
#Drawing the learning curve for TD method   
    def learning_curve_td (self, num_episode, num_curve):
        mean_sum = np.zeros((num_curve, num_episode))
        mean_plus_sum = np.copy(mean_sum)
        mean_minus_sum = np.copy(mean_sum)
        for i_curve in range (num_curve):
            mean, mean_plus_stdv, mean_minus_stdv = self.policy_iteration_td(num_episode)
            mean_sum[i_curve, :] = mean
            mean_plus_sum[i_curve, :] = mean_plus_stdv
            mean_minus_sum[i_curve, :] = mean_minus_stdv
            if((i_curve+1) % 10 == 0):
                print("learning curves:{}".format(i_curve+1))
        mean_plot = sum(mean_sum)/num_curve
        mean_plus_plot = sum(mean_plus_sum)/num_curve
        mean_minus_plot = sum(mean_minus_sum)/num_curve       
        x_axis = np.arange(1, num_episode+1, 1)
        plt.figure(figsize=(10,6))
        plt.subplot(1, 2, 1)
        plt.plot(x_axis, mean_plot, label='Mean')
        plt.legend(loc = 'lower right')
        plt.xlabel("Number of Episodes")
        plt.ylabel("Mean Reward")
        plt.title("Learning curve for a=0.05 & e=0.05")
        plt.grid(linestyle = ':')
        plt.subplot(1, 2, 2)
        plt.plot(x_axis, mean_plot, color='r', linestyle='-', label='Mean')
        plt.plot(x_axis, mean_plus_plot, color='b', linestyle=':', label='Mean+Stdv')
        plt.plot(x_axis, mean_minus_plot, color='g', linestyle=':', label='Mean-Stdv')      
        plt.legend(loc = 'lower right')
        plt.xlabel("Number of Episodes")
        plt.ylabel("Mean Reward & Mean Reward+/-Stdv")
        plt.grid(linestyle = ':')
        plt.show()
#Get a action given a state (for TD)        
    def get_action_td(self, state, policy):
        if(self.terminal[0, state]):
            return 0
        policy_p = np.copy(policy[state, :])
        action = np.random.choice([0, 1, 2, 3], size=1, replace=False, p=policy_p.ravel())
        action = int(action)
        return action
#Get the post state and reward given the action and state from get_action_td fuction
    def next_state_td(self, state, action):
        action_p = [0, 0, 0, 0]
        for i_action in range(4):
            if(i_action == action):
                action_p[i_action] = self.prop_desire
            else:
                action_p[i_action] = self.prop_oth
        movement = np.random.choice([0, 1, 2, 3], size=1, replace=False, p=action_p)
        movement = int(movement)
        post_state = self.neighbours[state,movement]
        post_state = int(post_state)
        if(post_state == 8):
            reward = 10
        elif(post_state == 21):
            reward = -100
        else:
            reward = -1
        return post_state, reward
#TD method to estimate the optimal solution. Note that here the returns are for poltting the learning curve.
    def policy_iteration_td(self, num_episode):
        starting_states = list(range(self.state_size))
        terminal_states = [8, 21]
        for del_state in terminal_states:
            starting_states.remove(del_state)
        Q = np.zeros((self.state_size, self.action_size))
        alpha = 0.00001
        policy = self.policy_improvement(Q)
        episode_return = []
        mean = []
        stdv = []
        for i_episode in range(1, num_episode+1):
            G = 0
            state = random.choice(starting_states)
            policy = self.policy_improvement(Q)
            action = self.get_action_td(state, policy)
            while not (self.terminal[0, state]):                
                post_state, reward = self.next_state_td(state, action)
                policy = self.policy_improvement(Q)
                post_action = self.get_action_td(post_state, policy)
                Q[state, action] = Q[state, action] + alpha*(reward + self.discount * Q[post_state, post_action] - Q[state, action])
                state = post_state
                action = post_action
                G = self.discount * G + reward
            episode_return.append(G)
            mean.append(G)
            stdv.append(np.std(episode_return))
        mean_np = np.array(mean)
        std_np = np.array(stdv)
        mean_plus_stdv = mean_np + std_np
        mean_minus_stdv = mean_np - std_np
        return mean_np, mean_plus_stdv, mean_minus_stdv
#Using TD method to evaluate a policy
    def policy_evaluation_td(self, policy):
        starting_states = list(range(self.state_size))
        terminal_states = [8, 21]
        for del_state in terminal_states:
            starting_states.remove(del_state)
        V = np.zeros(self.state_size)
        alpha =0.05
        num_episode = 1000
        for i_episode in range(1, num_episode+1):
            state = random.choice(starting_states)
            while not (self.terminal[0, state]):
                action = self.get_action_td(state, policy)
                post_state, reward = self.next_state_td(state, action)
                V[state] = V[state] + alpha * (reward + self.discount * V[post_state] - V[state])
                state = post_state
        return V
#A simple function to calculate the roor mean square error
    def rms (self, a, b):
        return np.sqrt(((a - b) ** 2).mean())
#For drawing two error plots 
    def estimation_error(self, num_episode, num_curve = 1):
        starting_states = list(range(self.state_size))
        terminal_states = [8, 21]
        for del_state in terminal_states:
            starting_states.remove(del_state) 
        alpha = 0.05
#The commented part can be uncommented to draw a different plot
        error_mc = np.zeros((num_curve, num_episode))
        error_td = np.copy(error_mc)
        # episode_reward_mc = np.copy(error_mc)
        # episode_reward_td = np.copy(error_td)
        true_value = self.get_true_value()
        #For a number of runs:        
        for i_curve in range (num_curve):
            Qmc = np.zeros((self.state_size, self.action_size))
            Qtd = np.copy(Qmc)
            Vmc = np.zeros(self.state_size)
            Vtd = np.copy(Vmc)
            policy_td = self.policy_improvement(Qtd)
            policy_mc = np.copy(policy_td)
            #one episode start
            for i_episode in range(num_episode):
                if((i_episode+1) % 500 == 0):
                    print("num {}".format(i_episode+1))
                episode_mc = self.get_episode(starting_states, policy_mc)
                Gmc = 0 #the episode reward, counting backwards
                state_td = random.choice(starting_states)
                policy_td = self.policy_improvement(Qtd)
                action_td = self.get_action_td(state_td, policy_td)
                #for MC
                for i_state_mc in range(len(episode_mc)):
                    state_mc, action_mc, reward_mc = episode_mc [i_state_mc]
                    Gmc = self.discount * Gmc + reward_mc
                    Qmc[state_mc, action_mc] += alpha * (Gmc - Qmc[state_mc, action_mc])
                    Vmc[state_mc] += alpha * (Gmc - Vmc[state_mc])
                policy_mc = self.policy_improvement(Qmc)
                #for TD
                Gtd = 0 #the episode reward
                while not (self.terminal[0, state_td]):                
                    post_state_td, reward_td = self.next_state_td(state_td, action_td)
                    policy_td = self.policy_improvement(Qtd)
                    post_action_td = self.get_action_td(post_state_td, policy_td)
                    Qtd[state_td, action_td] = Qtd[state_td, action_td] + alpha*(reward_td + self.discount * Qtd[post_state_td, post_action_td] - Qtd[state_td, action_td])
                    Vtd[state_td] = Vtd[state_td] + alpha*(reward_td + self.discount * Vtd[post_state_td] - Vtd[state_td])
                    state_td = post_state_td
                    action_td = post_action_td
                    Gtd = self.discount * Gtd + reward_td
                # episode_reward_mc[i_curve, i_episode] = Gmc
                # episode_reward_td[i_curve, i_episode] = Gtd
                error_mc[i_curve, i_episode] = self.rms(true_value, Vmc)
                error_td[i_curve, i_episode] = self.rms(true_value, Vtd)
            if((i_curve+1) % 100 == 0):
                print("curve {}".format(i_curve+1))
        # reward_axis_mc = sum(episode_reward_mc) / num_curve
        # reward_axis_td = sum(episode_reward_td) / num_curve
        rms_mc = sum(error_mc) / num_curve
        rms_td = sum(error_td) / num_curve
        x_axis = np.arange(1, num_episode+1, 1) 
        fig = plt.figure(figsize=(10, 6))
        error_curve = fig.add_subplot(111)
        error_curve.plot(x_axis, rms_mc, color='blue', linestyle='-', label='Error for MC')
        error_curve.plot(x_axis, rms_td, color='red', linestyle='-', label='Error for TD')
        # plt.title(label="Average of {} curves".format(num_curve))
        plt.grid(linestyle=':')
        plt.legend(loc='upper right')
        plt.show()
        # error_curve = fig.add_subplot(111)
        # error_curve.scatter(reward_axis_mc, rms_mc, color='blue', alpha=0.5, label='Error for MC')
        # error_curve.scatter(reward_axis_td, rms_td, color='red', alpha=0.5, label='Error for TD')
        # plt.xlabel("Episode reward")
        # plt.ylabel("Root Mean Square Error")
        # plt.title(label="Average of {} curves".format(num_curve))
        # plt.grid(linestyle=':')
        # plt.legend(loc='upper right')
        # plt.show()
        
#print the value function in console
    def print_value (self, value):
        index = 0
        for i in range(36):
            if (i == 7 or i == 15 or i == 17 or i == 19 or i == 25 or i == 26 or i ==28):
                print('\t', 'obst', end = '' )
                if (i % 6 == 5):
                    print('')
                continue
            v = float(value[index])
            print('\t', round(v, 1), end = '')
            index+=1
            if (i % 6 == 5):
                print('')
        print('')
        return
#print the policy in console
    def print_policy (self, policy):
        index = 0
        for i in range(36):
            if (i == 7 or i == 15 or i == 17 or i == 19 or i == 25 or i == 26 or i ==28):
                print('\t\t', 'ob', end = '')
                if (i % 6 == 5):
                    print('')
                continue
            if (i == 9 or i == 27):
                print('\t\t', 'T', end = '')
                index += 1
                continue
            for j in range(policy.shape[1]):
                if (policy[index, j] > 0.5):
                    if(j == 0):
                        print('\t\t', '↑' ,end = '')
                        continue
                    if(j == 1):
                        print('\t\t', '→' ,end = '')
                        continue
                    if(j == 2):
                        print('\t\t', '↓' ,end = '')
                        continue
                    if(j == 3):
                        print('\t\t', '←' ,end = '')
                        continue
            index+=1
            if(i % 6 == 5):
                print('')
        print('')
        return
    
#main function starts here. Functions can be uncommented to execute
grid=GridWorld()
# V = grid.get_true_value()
# grid.print_value(V)
grid.learning_curve_td(1500, 1000)
# grid.estimation_error(1000,1000)
