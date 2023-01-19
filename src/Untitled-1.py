# %%
import sys
import numpy as np
from simulator import *
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output
import os



# %%
def main(x="",seed=0):
    #ソルバの初期化
    print(f"シード{seed}での処理を開始します")
    sim = simulator()
    np.random.seed(seed)
    num = 1000
    config = {
        "write_all": False,
        "write_down": False
    }
    for i in range(num):
        sim.clear_data()
        #経路をランダム生成
        route = np.random.randint(1, 50, (4, 2))
        if i%10 == 0:
            sim.simulate(route,config["write_down"],config["write_all"])
        else:
            sim.simulate(route)
        print(f"done {i+1}/{num}")
    sim.create_csv(f'test1000_{x}.csv')
    print("saved as " + f'test1000_{x}.csv')
    print(f"シード{seed}での処理が終了しました")
    return x




# %%
from concurrent import futures
import time
from multiprocessing import Process, freeze_support

if __name__ == '__main__':
    freeze_support()

    #並列処理
    with futures.ProcessPoolExecutor(max_workers=8) as executor:
        [print(i) for i in executor.map(main, range(10), range(10))]
