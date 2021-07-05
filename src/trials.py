import os, pandas, time, torch, numpy
from src import comparison

if __name__ == "__main__":

    save_to = 'D:\ETH\projects\morpho-learner\\res\\comparison\\'
    comparison.plot_number_of_clusters('drugs', 300, save_to, filter_threshold=4)