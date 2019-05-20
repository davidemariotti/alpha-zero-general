import Arena
from MCTS import MCTS
from awari.AwariGame import AwariGame, display
from awari.keras.NNet import NNetWrapper as NNet
from awari.AwariPlayers import *
# to ask the oracle:
from awari.AwariLogic import Board
import sys
# from subprocess import check_output

import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir",  type=str, default="./temp/awari-keras/")
parser.add_argument("-f", "--file", type=str, default="best.pth.tar")
parser.add_argument("-n", "--number", type=int, default=2)
parser.add_argument("-m", "--mcts", type=int, default=200)
parser.add_argument("-c", "--cpuct", type=float, default=1.0)
parser.add_argument("-o", "--opponent", type=str, default="fop3")
parser.add_argument("-p", "--player", type=str, default="nn")
args = parser.parse_args()

print("directory: %s" % args.dir)
print("file: %s" % args.file)
print("number: %d" % args.number)
print("mcts: %d" % args.mcts)
print("cpuct: %f" % args.cpuct)
print("player: %s" % args.player)
print("opponent: %s" % args.opponent)

g = AwariGame()

class AwariNeuralNetPlayer():
    def __init__(self, game):
        self.game = game
        self.n1 = NNet(game)
        self.n1.load_checkpoint(args.dir, args.file)

    def play(self, board):
        args1 = dotdict({'numMCTSSims': args.mcts, 'cpuct': args.cpuct})
        mcts1 = MCTS(self.game, self.n1, args1)
        actions = mcts1.getActionProb(board, temp=1)
        select = np.argmax(actions)
        print('board: ', end="")
        print(board)
        print('action p-values: ' + str(actions))
        
        b = Board(6)
        b.pieces = np.copy(board)
        b.check_board(select, prefix = "nn: ")

        return select

# all players
def getPlayer(player):
    if player == "random":
        return RandomAwariPlayer(g).play
    elif player == "greedy":
        return GreedyAwariPlayer(g).play
    elif player == "human":
        return HumanAwariPlayer(g).play
    elif player == "oracle":
        return OracleAwariPlayer(g).play
    elif player == "fop2":
        return OracleAwariPlayer(g, 0.20, 15).play
    elif player == "fop25":
        return OracleAwariPlayer(g, 0.25, 15).play
    elif player == "fop3":
        return OracleAwariPlayer(g, 0.30, 15).play
    elif player == "fop4":
        return OracleAwariPlayer(g, 0.40, 15).play
    elif player == "fop5":
        return OracleAwariPlayer(g, 0.50, 15).play
    elif player == "fop6":
        return OracleAwariPlayer(g, 0.60, 15).play
    elif player == "fop7":
        return OracleAwariPlayer(g, 0.70, 15).play
    elif player == "fop8":
        return OracleAwariPlayer(g, 0.80, 15).play
    elif player == "mp2":
        return MinMaxAwariPlayer(g, 2).play
    elif player == "mp4":
        return MinMaxAwariPlayer(g, 4).play
    elif player == "mp6":
        return MinMaxAwariPlayer(g, 6).play
    elif player == "nn":
        return AwariNeuralNetPlayer(g).play
    else:
        print("Unkown player: " + player)
        return None

opponent = getPlayer(args.opponent)
if opponent == None:
    sys.exit(1)


player = getPlayer(args.player)
if player == None:
    sys.exit(1)


arena = Arena.Arena(player, opponent, g, display=display)
print("win/lost/draw", arena.playGames(args.number, verbose=True))
