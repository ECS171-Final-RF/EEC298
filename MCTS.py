# Written by Jingwei Wan
# Monte Carlo Searching Tree

import numpy as np
from NeuralNetworkTest import *


class MCTNode:
    def __init__(self, state, action, prob, Q):
        self.state = state
        self.action = action
        self.N = 0
        self.P = prob
        self.Q = Q
        self.sibling = None
        self.firstChild = None
        self.parent = None
        self.UCB = self.calUCB()

    # Initiate p vector based on the result calculated by NN
    def initPVector(self, pVector):
        self.pVector = pVector

    # Calculate UCB
    def calUCB(self):
        UCBfactor = 1.0
        self.UCB = self.Q + UCBfactor * self.P / (1 + self.N)
        return self.UCB

    # Node being visited
    def visited(self):
        self.N += 1





class MCT():
    def __init__(self, rootState, BOARD_SIZE, NN):
        self.rootState = rootState
        self.BOARD_SIZE = BOARD_SIZE
        self.NN = NN

    def buildTree(self, iterTime):
        self.root = MCTNode(state = self.rootState, action = None, prob = 1.0, Q = 0.0)
        # print(self.root.state)
        self.expandAll(self.root)
        pVector, V = self.NN.output(self.root.state)
        self.root.initPVector(pVector)

        for iter in range(iterTime):
            selectedNode = self.select(self.root)
            print(selectedNode.action)
            self.expandAll(selectedNode)
            # print(selectedNode)
            # Calculate p vector and v based on my NN
            pVector, v = self.NN.output(selectedNode.state)
            # pVector = pVector / np.sum(pVector)
            selectedNode.initPVector(pVector)

        # # Randomly choose one move
        # move = np.random.choice(self.BOARD_SIZE ** 2, p = pVector)
        # print(move)
        # move = [int(move / self.BOARD_SIZE), np.mod(move, self.BOARD_SIZE)]
        # print(move)







    # Expand one layer completely
    def expandAll(self, node):
        # Find out all possible moves
        possibleMoves = np.argwhere(node.state[3] == 0)
        # Expand a new layer with all possible moves
        tempNode = None
        for (i, move) in enumerate(possibleMoves):
            print("move:", move)
            nextState, reward, done, info = go_env.step(move)
            print(nextState[0])
            # Create a new node (prob and Q need to be calculated by the NN)
            newNode = MCTNode(state = nextState, action = move, prob = 1.0, Q = 0.0)
            if i == 0:
                node.firstChild = newNode
            else:
                tempNode.sibling = newNode
            tempNode = newNode

        # pivot = node.firstChild
        # while pivot != None:
        #     print(pivot.action)
        #     pivot = pivot.sibling


    # Select the leaf node with the largest UCB
    def select(self, root):
        if root.firstChild == None:
            # print(root)
            return root
        child = root.firstChild
        selectedNode = child
        while child != None:
            # print(child.UCB)
            if child.UCB > selectedNode.UCB:
                selectedNode = child
            child = child.sibling
        return self.select(selectedNode)



    # Build and evaluate a new node
    def evaluate(self):
        pass






















if __name__ == '__main__':
    import gym
    BOARD_SIZE = 5
    go_env = gym.make('gym_go:go-v0', size=BOARD_SIZE, reward_method='real')

    NNTest = MyNN(BOARD_SIZE = BOARD_SIZE)


    first_action = (2, 3)
    state, reward, done, info = go_env.step(first_action)

    firstMCT = MCT(state, BOARD_SIZE = 5, NN = NNTest)
    firstMCT.buildTree(iterTime = 10)
    # print(firstMCT.rootState)


