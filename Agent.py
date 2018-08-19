import gym
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import pickle
import time

env = gym.make("Taxi-v2")
env.render()

action_size = env.action_space.n
state_size = env.observation_space.n

#Hyperparameters
epsilon = 0.9                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability
decay_rate = 0.01             # Exponential decay rate for exploration prob


gamma = 0.618
alpha = 0.7
num_episodes = 50000
maxSteps = 99

QTable = np.zeros((state_size,action_size))


def QLearning():
    successfulEpisodes = 0
    for episode in range(num_episodes):
        steps = 0
        state = env.reset()
        isComplete = False

        while steps<maxSteps:
            steps+=1
            if random.uniform(0,1)<epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(QTable[state,:])

            newState, r, isComplete, _ = env.step(action)
            QTable[state,action] = QTable[state,action] + alpha*(r + gamma*np.max(QTable[newState,:]) - QTable[state,action])
            state = newState

            if isComplete == True:

                successfulEpisodes+=1
                print("Successful Episode " +  str(episode))
                break
            else:
                print("Episode Done " + str(episode))

        epsilon = epsilon * pow(2.71828, -(0.01 * (episode + 1)))
        if epsilon < min_epsilon:
            epsilon = min_epsilon

    print("Number of Successful Episodes" + str((successfulEpisodes*1.0)/(num_episodes)*1.0))
    f = open("QTable.dict","wb")
    pickle.dump(QTable,f)
    f.close()

def Explore():
    pickle_in = open("QTable.dict","rb")
    QTable = pickle.load(pickle_in)
    for episode in range(num_episodes):
        state = env.reset()
        steps = 0
        while steps < maxSteps:
            env.render()
            time.sleep(5)
            steps +=1
            action = np.argmax(QTable[state,:])
            newState, r, isComplete, _ = env.step(action)
            if isComplete == True:
                break
            else:
                state = newState

#Explore()
def QLearningNeuralNetwork():
    action_size = env.action_space.n
    state_size = env.observation_space.n

    # Hyperparameters
    epsilon = 0.9  # Exploration rate
    max_epsilon = 1.0  # Exploration probability at start
    min_epsilon = 0.01  # Minimum exploration probability
    decay_rate = 0.01  # Exponential decay rate for exploration prob

    gamma = 0.618
    alpha = 0.7
    num_episodes = 50000
    maxSteps = 99

    tf.reset_default_graph()


    #Establish the feed forward network
    inputs1 = tf.placeholder(shape=[1,state_size],dtype=tf.float32)
    W = tf.Variable(tf.random_uniform([state_size,6],0,0.01))
    Qout = tf.matmul(inputs1,W)
    predict = tf.argmax(Qout,1)

    #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
    nextQ = tf.placeholder(shape=[None,6],dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(nextQ - Qout))
    trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    updateModel = trainer.minimize(loss)


    init = tf.initialize_all_variables()

    #steps per episode
    stepList = []
    rewardList = []

    with tf.Session() as sess:
        for episode in range(num_episodes):
            sess.run(init)
            s = env.reset()
            print(s)
            steps = 0
            isComplete = False
            rAll = 0

            while steps<maxSteps:
                steps += 1

                a, allQ = sess.run([predict, Qout], feed_dict={inputs1: np.identity(state_size)[s:s + 1]})
                if random.uniform(0,1) < epsilon:
                    a = env.action_space.sample()


                s1, r, isComplete, _ = env.step(a)

                Q1 = sess.run(Qout, feed_dict={inputs1: np.identity(state_size)[s1:s1 + 1]})


                print((allQ.shape[0]))

                maxQ1 = np.max(Q1)
                targetQ = allQ

                targetQ[a] = r + gamma * maxQ1

                _, W1 = sess.run([updateModel, W], feed_dict={inputs1: np.identity(25)[s:s + 1], nextQ: targetQ})
                rAll += r
                s = s1
                if isComplete == True:
                    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
                    print("Successful Completed Episode : " + str(episode))

                    break

            print("Completed Episode : " + str(episode))
            stepList.append(steps)
            rewardList.append(rAll)

    print("Percent of succesful episodes: " + str(sum(rewardList)/num_episodes))
    np.save("rewardList.npy",rewardList)
    np.save("stepList.npy",stepList)

def QLearningFrozenLake():
    env = gym.make('FrozenLake-v0')
    tf.reset_default_graph()

    inputs1 = tf.placeholder(shape=[1, 16], dtype=tf.float32)
    W = tf.Variable(tf.random_uniform([16, 4], 0, 0.01))
    Qout = tf.matmul(inputs1, W)
    predict = tf.argmax(Qout, 1)

    # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
    nextQ = tf.placeholder(shape=[1, 4], dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(nextQ - Qout))
    trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    updateModel = trainer.minimize(loss)

    init = tf.initialize_all_variables()

    # Set learning parameters
    y = .99
    e = 0.1
    num_episodes = 2000
    # create lists to contain total rewards and steps per episode
    jList = []
    rList = []
    with tf.Session() as sess:
        sess.run(init)
        for i in range(num_episodes):
            # Reset environment and get first new observation
            s = env.reset()
            rAll = 0
            d = False
            j = 0
            # The Q-Network
            while j < 99:
                j += 1
                # Choose an action by greedily (with e chance of random action) from the Q-network
                a, allQ = sess.run([predict, Qout], feed_dict={inputs1: np.identity(16)[s:s + 1]})
                if np.random.rand(1) < e:
                    a[0] = env.action_space.sample()
                # Get new state and reward from environment
                s1, r, d, _ = env.step(a[0])
                # Obtain the Q' values by feeding the new state through our network
                Q1 = sess.run(Qout, feed_dict={inputs1: np.identity(16)[s1:s1 + 1]})
                # Obtain maxQ' and set our target value for chosen action.
                maxQ1 = np.max(Q1)
                targetQ = allQ
                targetQ[0, a[0]] = r + y * maxQ1
                # Train our network using target and predicted Q values
                _, W1 = sess.run([updateModel, W], feed_dict={inputs1: np.identity(16)[s:s + 1], nextQ: targetQ})
                rAll += r
                s = s1
                print("hi")
                if d == True:
                    # Reduce chance of random action as we train the model.
                    e = 1. / ((i / 50) + 10)
                    break
            jList.append(j)
            rList.append(rAll)
    print("Percent of succesful episodes: " + str(sum(rList) / num_episodes) + "%")




QLearningFrozenLake()



