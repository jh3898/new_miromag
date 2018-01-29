"""
use finte differences with a center difference methos in space and Crask-Nicolson method in time

"""

import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

plt.ion()

from operator1d import OperatorFiniteDiff1DPeriodic


class Solver1D(object):
    _Operator = OperatorFiniteDiff1DPeriodic

    def __init__(self, dt, nu, U, shape, lengths=None):
        self.dt = float(dt)

        self.U = float(U)
        self.nu = float(nu)
        self.oper = self._Operator(shape, lengths=lengths)
        self.L = self.linear_operator()
        self.A = self.oper.identity() - self.dt / 2 * self.L

        # initial condition
        self.t = 0.
        self._init_field()

        self._init_plot()

    def _init_field(self):
        self.s = np.exp(-(10 * (self.oper.xs - self.oper.Lx / 2)) ** 2)

    def linear_operator(self):
        return -self.U * self.oper.sparse_px + self.nu * self.oper.sparse_pxx

    def right_hand_side(self, s=None):
        if s is None:
            s = self.s
        return s.ravel() + self.dt / 2 * self.L.dot(s.flat)

    def one_time_step(self):
        self.s = spsolve(self.A, self.right_hand_side())
        self.s = self.s.reshape(self.oper.shape)
        self.t += self.dt

    def start(self, t_end=1.):
        while self.t < t_end:
            self.one_time_step()
            self._update_plot()

    def _init_plot(self):
        plt.figure()
        ax = plt.gca()
        ax.set_xlabel('x')
        ax.set_ylabel('s')
        ax.set_ylim(-0.1, 1)
        self.ax = ax
        self.line, = ax.plot(self.oper.xs, self.s)
        plt.show()

    def _update_plot(self):
        self.line.set_data(self.oper.xs, self.s)
        self.ax.figure.canvas.draw()
        plt.show()

if __name__ == '__main__':
    dt = 0.01
    U = 1.
    Lx = 1.
    nx = 400
    nu = 0.

    sim = Solver1D(dt, nu, U, [nx], [Lx])
    print('holycrap')
    sim.start(1)
