import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from solver import *
import os


class simulator:
    # 指定された方向にフリックするプログラム
    def __init__(self, N):
        self.df = pd.DataFrame()
        self.N = N
        self.size = self.N + 2
        self.dt = 0.1
        # 拡散項
        self.diff = 0.00001
        # 粘性係数
        self.visc = 0.01
        # 力の固定値
        self.force = 20.0
        self.source = 200.0
        # #tmp_pos_x,y から pos_x,y に移動するときにu,v を変更している
        self.tmp_pos_x = 0.0
        self.tmp_pos_y = 0.0
        self.pos_x = 0.0
        self.pos_y = 0.0

        # 横速度
        self.u = np.zeros((self.size, self.size), np.float64)  # velocity
        self.u_prev = np.zeros((self.size, self.size), np.float64)
        # 縦速度
        self.v = np.zeros((self.size, self.size), np.float64)  # velocity
        self.v_prev = np.zeros((self.size, self.size), np.float64)
        # 密度
        self.dens = np.zeros((self.size, self.size), np.float64)  # density
        self.dens_prev = np.zeros((self.size, self.size), np.float64)

    def clear_data(self):
        """clear_data."""
        self.u = np.zeros((self.size, self.size), np.float64)  # velocity
        self.u_prev = np.zeros((self.size, self.size), np.float64)
        # 縦速度
        self.v = np.zeros((self.size, self.size), np.float64)  # velocity
        self.v_prev = np.zeros((self.size, self.size), np.float64)
        # 密度
        self.dens = np.zeros((self.size, self.size), np.float64)  # density
        self.dens_prev = np.zeros((self.size, self.size), np.float64)

    def move(self, prev_x, prev_y, x, y):
        """状況のアップデート"""
        #  move(dens_prev, u_prev, v_prev,int(cur_x),int(cur_y),int(new_x),int(new_y))

        self.dens_prev[0:self.size, 0:self.size] = 0.0
        self.u_prev[0:self.size, 0:self.size] = 0.0
        self.v_prev[0:self.size, 0:self.size] = 0.0

        dx = x - prev_x
        dy = y - prev_y

        # 流入も移動もない場合，そもそも関数が呼ばれない

        # 更新する座標を選択
        if x < 1 or x > self.N or y < 1 or y > self.N:
            return

        move = True
        pour = True
        if move:
            wave_size = 2
            for delta_x in range(1-wave_size, wave_size-1):
                newx = x + delta_x
                for delta_y in range(1-wave_size, wave_size-1):
                    newy = y + delta_y
                    if newx > 1 and newx < self.N and newy > 1 and newy < self.N:
                        self.u_prev[newx, newy] = self.force * (dx)
                        # TODO -dyに戻す←戻さない！！
                        self.v_prev[newx, newy] = self.force * (dy)
        if pour:
            self.dens[x, y] += self.source
            wave_size = 2
            # 3マス四方が領域内であれば
            for delta_x in range(1-wave_size, wave_size-1):
                newx = x + delta_x
                for delta_y in range(1-wave_size, wave_size-1):
                    newy = y + delta_y
                    if newx > 1 and newx < self.N and newy > 1 and newy < self.N:
                        self.dens_prev[newx, newy] += self.source

    def calc_init_vec(self, dx, dy, v=1):
        # 長さ1の，xとyを求める
        if dx == 0:
            return (0, 2)

        tan = dy/dx
        rad = np.arctan2(dy, dx)
        ex = np.cos(rad) * v
        ey = np.sin(rad) * v

        return (ex, ey)

    def create_picture(self, dens, dirname, filename):
        # densの値を2倍
        display_dens = dens * 2
        # グレースケールとして扱い，表示
        plt.imshow(display_dens, cmap='gray')
        # cntがintならば
        filepath = os.path.join(dirname, filename)
        plt.savefig(filepath)

    def save_df(self, route, res):
        # display_densを一次元配列に
        res_row = res.flatten()
        # routeを一次元配列に
        res_route = np.array(route).flatten()
        # routeを付け足して，dataframeにする
        res = np.concatenate([res_route, res_row])
        # dfに書き込む
        self.df = self.df.append(pd.Series(res), ignore_index=True)

    def simulate(self, route, write_down=False, write_all=False):
        # 経路の情報を受け取り、シミュレーション結果を返す
        l = len(route)
        # 時間を固定
        timestep = 30
        cnt = 0
        # ソルバを初期化
        self.clear_data()

        for i in (0, 2):
            start_x, start_y = route[i]
            goal_x, goal_y = route[i + 1]
            dx = goal_x - start_x
            dy = goal_y - start_y
            route_len = np.sqrt((start_x - goal_x) ** 2 +
                                (start_y - goal_y) ** 2)
            # step_len = route_len / timestep
            velocity = 1
            step_vec = self.calc_init_vec(dx, dy, velocity)
            timestep = int(route_len / velocity)
            cur_x = start_x
            cur_y = start_y
            for j in range(timestep):
                cnt += 1
                new_x = cur_x + step_vec[0]
                new_y = cur_y + step_vec[1]
                self.move(int(cur_x), int(cur_y), int(new_x), int(new_y))
                vel_step(self.N, self.u, self.v, self.u_prev,
                         self.v_prev, self.visc, self.dt)
                dens_step(self.N, self.dens, self.dens_prev,
                          self.u, self.v, self.diff, self.dt)
                cur_x = new_x
                cur_y = new_y
                if write_all:
                    dirname = f"res\{write_all}"
                    filename = f"dens{cnt:03d}.png"
                    self.create_picture(self.dens, dirname, filename)
                    # 1行消して，print
                    print(f"\r{cnt} done", end="")
            # 100timestep待つ
            if i == 2:
                break
            for j in range(10):
                cnt += 1
                vel_step(self.N, self.u, self.v, self.u_prev,
                         self.v_prev, self.visc, self.dt)
                dens_step(self.N, self.dens, self.dens_prev,
                          self.u, self.v, self.diff, self.dt)
                self.u_prev = self.u.copy()
                self.v_prev = self.v.copy()
                if write_all:
                    dirname = f"res\{write_all}"
                    filename = f"dens{cnt:03d}.png"
                    self.create_picture(self.dens, dirname, filename)
                    print(f"\r{cnt} done", end="")

        self.save_df(route, self.dens)
        if write_down and not write_all:
            filename = f"dens{cnt:03d}.png"
            if type(write_down) == int:
                dirname = f"res/number"
            else:
                dirname = f"res/final"

            # write_downにはcntが入っている
            self.create_picture(self.dens, dirname, filename)

    def create_csv(self, filename):
        self.df.to_csv(filename)


if __name__ == "__main__":
    # ソルバの初期化
    sim = simulator(30)
    np.random.seed(0)
    num = 1
    config = {
        "write_all": "True",
        "write_down": False
    }
    for i in range(num):
        sim.clear_data()
        # 経路をランダム生成
        route = np.random.randint(1, 50, (4, 2))
        print(route)
        if i % 10 == 0:
            sim.simulate(route, config["write_down"], config["write_all"])
        else:
            sim.simulate(route)
        print(f"done {i+1}/{num}")
    sim.create_csv(f'test1000_{-1}.csv')
    print("saved as " + f'test1000_{-1}.csv')
