"""
This module provides a more fine tuned time estimation method that could
(attempt to) adapt to varying dt.
"""
# TODO: Use robust regression for linear and exponential algorithms.

import numpy as np
import time


class EstimateTime():

    def __init__(self, start_time, total, mode):
        self.start_time = start_time
        self.last_time = start_time
        self.iteration = 0
        self.total = total
        self.__elapsed = 0
        self.dt_list = []
        self.est_time = 0

        # If a mode is specified, the auto select function will be overridden.
        if mode is not 'auto':
            self.override = True
            self.min_key = mode
        else:
            self.override = False
            self.min_key = 'average'

        # Containers for the storage of discrepancies between the estimated
        #  time and the actual dt.
        self.lin_err = [0] * 40
        self.average_err = [0] * 40

        # Containers for the storage of predicted dt's.
        self.lin_est_next = [0] * 8
        self.average_est_next = [0] * 8

    def stop_watch(self):
        self.iteration += 1
        self.__elapsed = time.time() - self.start_time
        dt = time.time() - self.last_time
        self.last_time = time.time()
        self.dt_list.append(dt)
        if self.total >= 200 and self.total < 800:
            if len(self.dt_list) > 200:
                self.dt_list.pop(0)
        elif self.total >= 800:
            if self.iteration > self.total / 4:
                self.dt_list.pop(0)

    def est(self):
        """
        Returns the estimated time in seconds. In the current version,
        five algorithms are compared and the best one will be chosen
        for the estimation.
        """
        self.min_key = 'average'
        if self.iteration >= 3:
            # Poll estimated times from different algorithms
            average_est_time = self.average_est()
            lin_est_time = self.lin_est()

            # Record discrepancies between the estimated delta t's and the
            #  actual delta t.
            if self.iteration > 8:
                self.err_rec()

            # Review the choice of algorithm after every 15 jobs and switch
            #  to a better one if necessary.
            if not self.override:
                if self.iteration % 15 == 0 and self.iteration > 8:
                    self.least_err()

            # Return the time associated with the algorithm that offers the
            #  highest accuracy.
            if self.min_key is 'average':
                est_time = average_est_time
            elif self.min_key is 'lin':
                est_time = lin_est_time

            est_time = int(round(est_time))
        else:
            est_time = 0

        # Bypasses negative estimates occasionally generated by the linear
        #  algorithm and huge numbers occasionally generated by the positive
        #  exponential algorithm. 3.2e7 is a little over a year.
        if est_time < 0:
            est_time = self.est_time
            if not self.override:
                self.min_key = 'average'
        else:
            self.est_time = est_time

        return est_time

    def err_rec(self):
        """
        Records the errors of the estimated time for each algorithm.
        """
        last_dt_avg = self.dt_list[-8]

        # Records the discrepancies
        self.lin_err.append(abs(self.lin_est_next[0] - last_dt_avg))
        self.average_err.append(abs(self.average_est_next[0] - last_dt_avg))

        # Removes the oldest entry
        self.lin_err.pop(0)
        self.average_err.pop(0)

    def least_err(self):
        """
        Looks into the error records of all the algorithms and selects the
        one with the smallest error.
        """
        errs = {
            "lin": abs(sum(self.lin_err)),
            "average": abs(sum(self.average_err)),
        }
        self.min_key = min(errs, key=errs.get)

    def average_est(self):
        def reject_outliers(data, m=2.5):
            d = np.abs(data - np.median(data))
            mdev = np.median(d)
            s = d / mdev if mdev else 0.
            return data[s < m]

        dt_list = reject_outliers(np.array(self.dt_list))
        const = np.sum(dt_list) / len(dt_list)
        est_time = const * (self.total - self.iteration)

        est_time_next = const
        self.average_est_next.append(est_time_next)
        self.average_est_next.pop(0)
        return est_time

    def least_sq_fit(self, M, y):
        M = np.matrix(M)
        MTM = M.T * M
        v = MTM.I * M.T * y
        return v

    def linv(self):
        """
        Calculates the coefficients of model functions for linear
        least-squares best fit for the y = ax + b model.
        """
        def reject_outliers(data, m=2.5):
            d = np.abs(data - np.median(data))
            mdev = np.median(d)
            s = d / mdev if mdev else 0.
            return data[s < m]

        dt_list = reject_outliers(np.array(self.dt_list))
        M = np.empty([len(dt_list), 2])
        M[:, 0] = np.ones([len(dt_list)])
        M[:, 1] = np.arange(self.iteration, len(dt_list) + self.iteration)
        y = np.matrix(dt_list).T
        lin_v = self.least_sq_fit(M, y)
        return lin_v

    def lin_est(self):
        lin_v = self.linv()
        # Find the time estimate by integrating over the best fit function
        #  from the current point to the last point
        est_time = lin_v[1, 0] * (self.total**2 - self.iteration**2) / 2
        est_time += lin_v[0, 0] * (self.total - self.iteration)

        # Find the time estimate for the job 6 points in the future.
        #  For evaluation of the accuracy of the function.
        est_time_next = lin_v[1, 0] * ((self.iteration + 8)**2 - (self.iteration + 7)**2) / 2
        est_time_next += lin_v[0, 0]
        self.lin_est_next.append(est_time_next)
        self.lin_est_next.pop(0)
        return est_time
