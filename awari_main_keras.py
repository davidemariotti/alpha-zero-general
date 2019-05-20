from Coach import Coach
from awari.AwariGame import AwariGame
from awari.keras.NNet import NNetWrapper as nn
# from awari.tensorflow.NNet import NNetWrapper as nn
from utils import *
import time

args = dotdict({
    'numIters': 20,
    'numEps': 20,
    'tempThreshold': 400,
    'updateThreshold': 0.54,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 50,
    'arenaCompare': 20,
    'cpuct': 1,

    'checkpoint': './temp/awari-keras/',
    #'load_model': False,
    'load_model': True,
    'load_folder_file': ('./temp/awari-keras/', 'checkpoint_snapshot.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})

if __name__=="__main__":
    start = time.time()
    g = AwariGame()
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)

    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()

    c.learn()
    end  = time.time()
    print('total time: ')
    print(end - start)

