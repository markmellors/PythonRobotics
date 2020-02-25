"""

Path tracking simulation with pure pursuit steering control and PID speed control.

author: Atsushi Sakai (@Atsushi_twi)
        Guillaume Jacquenot (@Gjacquenot)

"""
import numpy as np
import math
import matplotlib.pyplot as plt


k = 0  # look forward gain
Lfc = 0.3  # look-ahead distance
Kp = 7  # speed proportional gain
dt = 0.05  # [s]
L = 0.1  # [m] wheel base of vehicle


show_animation = True


blindMaze = [[0.0, 0.0],
             [0.0, 0.3],
             [0.0, 0.6],
             [0.0, 0.9],
             [0.0, 1.1],
             [0.3, 1.1],
             [0.5, 1.1],
             [0.7, 1.1],
             [1.0, 1.1],
             [1.1, 1.0],
             [1, 0.8],
             [1, 0.5],
             [1, 0.2],
             [1.1, 0.1],
             [1.3, 0.2],
             [1.6, 0.2],
             [2, 0.2],
             [2.1, 0.3],
             [2, 0.5],
             [2, 0.8],
             [2, 1.1],
             [2.1, 1.2],
             [2.3, 1.1],
             [2.6, 1.1],
             [2.9, 1.1],
             [3.2, 1.1],
             [3.6, 1.1]]

class State:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.rear_x = self.x - ((L / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((L / 2) * math.sin(self.yaw))

    def update(self, a, delta):

        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw += self.v / L * math.tan(delta) * dt
        self.v += a * dt
        self.rear_x = self.x - ((L / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((L / 2) * math.sin(self.yaw))

    def calc_distance(self, point_x, point_y):

        dx = self.rear_x - point_x
        dy = self.rear_y - point_y
        return math.hypot(dx, dy)


class States:

    def __init__(self):
        self.x = []
        self.y = []
        self.yaw = []
        self.v = []
        self.t = []

    def append(self, t , state):
        self.x.append(state.x)
        self.y.append(state.y)
        self.yaw.append(state.yaw)
        self.v.append(state.v)
        self.t.append(t)


def PIDControl(target, current):
    a = Kp * (target - current)

    return a


class Trajectory:
    def __init__(self, cx, cy):
        self.cx = cx
        self.cy = cy
        self.old_nearest_point_index = None

    def search_target_index(self, state):
        if self.old_nearest_point_index is None:
            # search nearest point index
            dx = [state.rear_x - icx for icx in self.cx]
            dy = [state.rear_y - icy for icy in self.cy]
            d = np.hypot(dx, dy)
            ind = np.argmin(d)
            self.old_nearest_point_index = ind
        else:
            ind = self.old_nearest_point_index
            distance_this_index = state.calc_distance(self.cx[ind], self.cy[ind])
            while True:
                ind = ind + 1 if (ind + 1) < len(self.cx) else ind
                distance_next_index = state.calc_distance(self.cx[ind], self.cy[ind])
                if distance_this_index < distance_next_index:
                    ind = ind-1
                    break
                distance_this_index = distance_next_index
            self.old_nearest_point_index = ind

        L = 0.0

        Lf = k * state.v + Lfc

        # search look ahead target point index
        while Lf > L and (ind + 1) < len(self.cx):
            L = state.calc_distance(self.cx[ind], self.cy[ind])
            ind += 1
        return ind


def pure_pursuit_control(state, trajectory, pind):

    ind = trajectory.search_target_index(state)

    if pind >= ind:
        ind = pind

    if ind < len(trajectory.cx):
        tx = trajectory.cx[ind]
        ty = trajectory.cy[ind]
    else:
        tx = trajectory.cx[-1]
        ty = trajectory.cy[-1]
        ind = len(trajectory.cx) - 1

    alpha = math.atan2(ty - state.rear_y, tx - state.rear_x) - state.yaw

    Lf = k * state.v + Lfc

    delta = math.atan2(2.0 * L * math.sin(alpha) / Lf, 1.0)

    return delta, ind

def addArena():
    addRect(-0.25, -0.25, 3, 1.8)
    addRect(0.25, -0.25, 0.5, 1)
    addRect(1.25, 0.55, 0.5, 1)
    addRect(2.25, -0.25, 0.5, 1)

def addRect(x, y, w, h):
    p = plt.Rectangle((x,y),w,h,fill=False)
    ax = plt.gca()
    ax.add_patch(p)

def plot_arrow(x, y, yaw, length=0.1, width=0.05, fc="r", ec="k"):
    """
    Plot arrow
    """

    if not isinstance(x, float):
        for ix, iy, iyaw in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)


def main():
    #  target course
    cx = [row[0] for row in blindMaze]
    cy = [row[1] for row in blindMaze]

    target_speed = 2  # [m/s]

    T = 100.0  # max simulation time

    # initial state
    state = State(x=-0.0, y=0, yaw=1.57, v=0.0)

    lastIndex = len(cx) - 1
    time = 0.0
    states = States()
    states.append(time, state)
    trajectory = Trajectory(cx, cy)
    target_ind = trajectory.search_target_index(state)

    while T >= time and lastIndex > target_ind:
        ai = PIDControl(target_speed, state.v)
        di, target_ind = pure_pursuit_control(state, trajectory, target_ind)
        state.update(ai, di)

        time += dt
        states.append(time, state)
        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            addArena()
            plot_arrow(state.x, state.y, state.yaw)
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(states.x, states.y, "-b", label="trajectory")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            plt.axis("equal")
            plt.grid(True)
            plt.title("Speed[km/h]:" + str(state.v * 3.6)[:4])
            plt.pause(0.001)

    # Test
    assert lastIndex >= target_ind, "Cannot goal"

    if show_animation:  # pragma: no cover
        plt.cla()
        addArena()
        plt.plot(cx, cy, ".r", label="course")
        plt.plot(states.x, states.y, "--b", label="trajectory")
        plt.legend()
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.axis("equal")
        plt.grid(True)

        #plt.subplots(1)
        #plt.plot(states.t, [iv * 3.6 for iv in states.v], "-r")
        #plt.xlabel("Time[s]")
        #plt.ylabel("Speed[km/h]")
        #plt.grid(True)
        plt.show()


if __name__ == '__main__':
    print("Pure pursuit path tracking simulation start")
    main()
