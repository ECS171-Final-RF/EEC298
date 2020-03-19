# Written by Jingwei Wan
import numpy as np


class MyNN():
    def __init__(self, BOARD_SIZE):
        self.BOARD_SIZE = BOARD_SIZE

    def output(self, state):
        pVector = 1.0 / self.BOARD_SIZE ** 2 * np.ones(self.BOARD_SIZE ** 2)
        v = 1
        return pVector, v