#lunah landah example
#this example doesn't use target network
'''
Source: https://www.youtube.com/watch?v=wc-FxNENg9U
Deep Q Network is model-free
bootstrapped = determines actions either randomly or from prior states
off-policy: one policy generates explore actions (epsilon) and updates actions of exploit policy (greedy)
deep q network doesn't work for continuous action spaces

stick stuff into two main classes and give some basic functionality
agent isn't a deep q, it has a deep q
agent has memory, network doesn't
'''

import torch as T #base package
import torch.nn as nn #layer handling
import torch.nn.functional as F #activation function
import torch.optim as optim #adam optimizer
import numpy as np #always good to have


class DeepQNetwork(nn.Module): #class must derive from nn.Module
    #inherits optimization parameters and backpropagation
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        '''
        lr = learn rate
        input_dims = dimensions of input data
        fc1_dims = first layer dims (fully connected)
        fc2_dims = second layer dims
        n_actions = possible actions that agent can take
        '''
        #assign attributes to class
        super(DeepQNetwork, self).__init__() #calls constructor for parent class
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        
        #model layers 
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims) #*input_dims unpacks the variable, fc1_dims is layer output
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, n_actions) #outputs n_actions number of actions
        

        #model optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        #loss function
        self.loss = nn.MSELoss() #mean square error loss. Not using the Belman equation

        #set the GPU
        self.device = T.device('cuda:0' if T.cuda.is_available else 'cpu')
        print(self.device)
        self.to(self.device)

    def forward(self, state): #forward propagation handling, eg. define activation
        x = F.relu(self.fc1(state)) #pass state into first layer and actiavte with relu function
        x = F.relu(self.fc2(x)) #pass the transformed state into the second layer and activate with relu
        actions = self.fc3(x) #finally pass to third fully connected layer

        return actions

class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, 
            n_actions, action_space, max_mem_size=1000000, eps_end=0.01, eps_dec=5e-4):
        '''
        gamma = determines weighting and future rewards (long-term vs. short-term)
        epsilon = determines how often agent explores vs. exploits
        lr = learning rate
        input_dims = dimensions of input data, gets passed to DeepQNetwork
        batch_size = how many observations per batch
        n_actions = actions the agent can take                      #need to remove
        max_mem_size = maximum number of observations #renamed below
        eps_end = 'epsilon end,' ending value of epsilon #renamed below
        eps_dec = rate at which epsilon decays (this example uses linear decay)
        '''

        #assign attributes to class
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end                                  #name change
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = action_space
        self.mem_size = max_mem_size                            #name change
        self.batch_size = batch_size
        self.mem_cntr = 0 #position of first available memory
        # self.input_dims = input_dims
        # self.n_actions = n_actions #not part of example

        #define the q-network evaluation model
        #need to remove n_actions
        #import ipdb; ipdb.set_trace()
        self.Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims, 
            fc1_dims=256, fc2_dims=256)
        print(self.Q_eval.fc3.out_features)

        
        #memory handling
        '''
        this example uses named arrays for the replay buffer
        this allows for named memory calls (doesn't avoid indexed calls, see store_transition)
        data type is important because it determines level of precision, and torch cares a lot
        '''
        #initialize memory arrays
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32) #stores current state
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32) #stores the next state
        self.action_memory = np.zeros(self.mem_size) #stores past actions
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32) #stores rewards for action/state pairs
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool) #remembers which actions led to termination

    def store_transition(self, state, action, reward, state_, done):
        '''
        function to store transitions based on bellman equation
        reward(state*action) + gamma*v(state_)
        state = completed state
        action = completed action
        reward = any rewards
        state_ = next state
        done = any termination
        '''
        index = self.mem_cntr % self.mem_size #overwrites oldest memories, but wraps the array (pseudo-sorted)

        #write values to memory
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward  ##### This line was missing from my code, but not his
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        #increment your index
        self.mem_cntr += 1

    def choose_action(self, observation): #gets observation of current state of environment
        if np.random.random() > self.epsilon: #define exploitative behavior
            state = T.tensor([observation]).to(self.Q_eval.device) #turn observation to tensor and send to GPU/CPU
            actions = self.Q_eval.forward(state) #determine the current action
            action = T.argmax(actions).item() #return integer corresponding to maximal action
        else: #explorative behavior
            #action = np.random.choice(self.action_space)
            '''#take a random action
            #build a list of random inputs
            #5 float, 3 boolean'''
            action = self.random_action_generator()
        return action

            
    
    def random_action_generator(self):
        action_size = 8
        bool_space = 3
        all_action = action_size-bool_space
        
        #create a boolean array
        bool_start = all_action
        bool_end =  all_action+bool_space
        bool_choices = [True, False]

        #initialize action array
        action = np.zeros(all_action+bool_space, dtype = object)
        #add random values for all_action
        action[0:all_action] = np.random.uniform(-1.0, 1.0, size=all_action)
        #add random booleans for bool_space
        action[bool_start:bool_end] = np.random.choice(bool_choices, size=3)

        return action


    def learn(self):
        '''
        how will the agent learn from its experiences
        immediately, there is a dilemma: how will the agent choose an action if it doesn't have memory?
        one possibility: allow it to take random actions until its memory is full, then train
        another one: start learning as soon as batch size is filled
        this example uses batch size
        '''
        if self.mem_cntr < self.batch_size: #skip this function if batch size is not filled
            return
        
        self.Q_eval.optimizer.zero_grad() #zeroes out gradient on optimizer, pytorch specific

        max_mem = min(self.mem_cntr, self.mem_size) #determines current memory size
        batch = np.random.choice(max_mem, self.batch_size, replace=False) #select random memory
        #replace=False is there for case where there are few memories to select from
        
        batch_index = np.arange(self.batch_size, dtype=np.int32) #for proper array slicing

        #get memory batches, convert to tensors, and send to GPU/CPU
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch] #gets set of actions corresponding to batch choice
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        

        #use loss function to encourage agent to take maximal actions
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch] #get values of actions that were taken and 'dereferences'
        q_next = self.Q_eval.forward(new_state_batch) #gets estimate of next state, this is where you'd usually use target network
        q_next[terminal_batch] = 0.0 #values of terminal states are set to zero (idk why)

        #This is the direction in which we want to update our estimates
        #dim=1 for action dimension, max function returns value as well as index, where [0] is value
        #gamma is discount factor (short vs long term)
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0] 

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device) #calculate loss and send to GPU/CPU
        loss.backward() #back-propagate the loss
        self.Q_eval.optimizer.step() #step your optimizer

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min #epsilon decay

