# Written by Jingwei Wan
# Monte Carlo Searching Tree

import numpy as np
# from NeuralNetworkTest import *
from NeuralNetworkTest_Mingye import *


class MCTNode:
    def __init__(self, state, action, QSum, P):
        self.state = state
        self.action = action
        self.N = 0
        self.QSum = QSum
        self.sibling = None
        self.firstChild = None
        self.parent = None
        self.P = P
        self.UCB = self.calUCB()
        self.childNum = 0

    # Calculate UCB
    def calUCB(self):
        UCBfactor = 0.2
        if self.N == 0:
            self.UCB = self.QSum + UCBfactor * self.P
        else:
            self.UCB = self.QSum * 1.0 / self.N + UCBfactor * self.P / (1 + self.N)
        return self.UCB

    # Node being visited
    def visited(self):
        self.N += 1

    def showNodeInfo(self):
        print("black stone position:", np.argwhere(self.state[0] == 1))
        print("white stone position:", np.argwhere(self.state[1] == 1))
        if self.N == 0:
            print("action:", self.action, ", N:", self.N, ", QSum:", self.QSum, ", Q:", self.QSum,", P:", self.P, ", UCB:", self.UCB, "\n")
        else:
            print("action:", self.action, ", N:", self.N, ", QSum:", self.QSum, ", Q:", self.QSum * 1.0 / self.N,", P:", self.P, ", UCB:", self.UCB, "\n")





class MCT():
    def __init__(self, rootState, BOARD_SIZE, NN):
        self.rootState = rootState
        self.BOARD_SIZE = BOARD_SIZE
        self.NN = NN

    def buildTree(self, iterTime):
        self.root = MCTNode(state = self.rootState, action = None, QSum = 0.0, P = 1.0)
        # print(self.root.state)
        pVector, v = self.NN.output(self.root.state)
        self.expandAll(self.root, pVector)

        epsilon = 0.2

        for iter in range(iterTime):
            print("iteration time:", iter)

            r = np.random.random()
            if r < epsilon:
                selectedNode = self.selectRandom(self.root)
            else:
                selectedNode = self.selectUCB(self.root)

            print(selectedNode.action)
            # print(selectedNode)
            # Calculate p vector and v based on my NN
            pVector, v = self.NN.output(selectedNode.state)
            self.expandAll(selectedNode, pVector)
            self.backPropagation(selectedNode, v)

        self.showAllLeaf(self.root)



        # # Randomly choose one move
        # move = np.random.choice(self.BOARD_SIZE ** 2, p = pVector)
        # print(move)
        # move = [int(move / self.BOARD_SIZE), np.mod(move, self.BOARD_SIZE)]
        # print(move)







    # Expand one layer completely
    def expandAll(self, node, pVector):
        # Find out all possible moves
        possibleMoves = np.argwhere(node.state[3] == 0)
        possibleMoves = np.append(possibleMoves, [[-1, -1]], axis = 0)
        node.childNum = len(possibleMoves)
        # print(possibleMoves)
        # print("possible moves:", possibleMoves)

        ########### Game ends ########################
        if possibleMoves.size == 1 and node.state[-1][0][0] == 0:
            nextState, reward, done, info = go_env.step_batch(node.state, None)
            print("reward:", reward, ", done:", done)
            newNode = MCTNode(state=nextState, action=None, QSum=reward, P=1.0)
            node.firstChild = newNode
            newNode.parent = node
            return

        elif possibleMoves.size == 1 and node.state[-1][0][0] == 1:
            return


        ##############################################

        # Expand a new layer with all possible moves
        # print(pVector)
        move1DIndex = np.array([int(self.BOARD_SIZE * possibleMoves[i][0] + possibleMoves[i][1]) for i in range(len(possibleMoves)) ])
        move1DIndex[-1] = -1
        # print(move1DIndex)
        possiblePVector = pVector[move1DIndex]
        possiblePVector = possiblePVector / np.sum(possiblePVector)

        # print(possiblePVector)

        tempNode = None
        for (i, move) in enumerate(possibleMoves):
            if i == len(possibleMoves) - 1 and node.state[-1][0][0] == 1:
                return
            if i == len(possibleMoves) - 1:
                move = None
            # print("move:", move)
            nextState, reward, done, info = go_env.step_batch(node.state, move)
            # print(np.argwhere(nextState[0] == 1))
            # print(np.argwhere(nextState[1] == 1))
            # Create a new node (prob and Q need to be calculated by the NN)
            newNode = MCTNode(state = nextState, action = move, QSum = 1.1, P = possiblePVector[i])
            if i == 0:
                node.firstChild = newNode
            else:
                tempNode.sibling = newNode
            newNode.parent = node
            tempNode = newNode





        # pivot = node.firstChild
        # while pivot != None:
        #     print(pivot.action)
        #     pivot = pivot.sibling


    # Select the leaf node with the largest UCB
    def selectUCB(self, root):
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
        return self.selectUCB(selectedNode)


    def selectRandom(self, root):
        # root.showNodeInfo()
        if root.firstChild == None:
            # print(root)
            return root
        child = root.firstChild
        selectedIndex = np.random.randint(root.childNum)
        # print('Random number:', selectedIndex)
        i = 0
        while child != None and i != selectedIndex:
            child = child.sibling
            i = i + 1
        selectedNode = child
        return self.selectRandom(selectedNode)





    # Backpropagation
    def backPropagation(self, node, v):
        node.N += 1
        node.QSum += v
        node.calUCB()
        # node.showNodeInfo()
        if node.parent == None:
            return node

        return self.backPropagation(node.parent, v)

    def showAllLeaf(self, node):
        leaf = node.firstChild
        print('\nAll leaves:\n')
        while(leaf != None):
            leaf.showNodeInfo()
            leaf = leaf.sibling





















if __name__ == '__main__':
    import gym
    BOARD_SIZE = 3
    go_env = gym.make('gym_go:go-v0', size=BOARD_SIZE, reward_method='real')
    goGame = go_env.gogame
    NNTest = MyNN(BOARD_SIZE = BOARD_SIZE)

    initial_state = go_env.reset()
    # print(initial_state)

    # first_action = (2, 3)
    # state, reward, done, info = go_env.step(first_action)

    firstMCT = MCT(initial_state, BOARD_SIZE = BOARD_SIZE, NN = NNTest)
    firstMCT.buildTree(iterTime = 1000)
    # print(firstMCT.rootState)


