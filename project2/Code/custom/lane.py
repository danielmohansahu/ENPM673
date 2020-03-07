"""Lane Class; used for tracking/probablistic estimate of lanes.
"""

import numpy as np
import scipy.stats

class Lane:
    """This class maintains a list of lines thought to be in the same lane.

    All analysis is performed on the slope of the lane. Since this changes
    over time we keep track of our confidence on the last few detections.
    """
    def __init__(self, seed):
        # initialize our list of matched lanes with a single expected lane.
        self.matches = [seed]
        self.rho = None # best estimate
        self.slopes = [self.slope(*seed)]
        self.matches_per_frame = [1]
        self.n_lanes = 1000
        self.min_frames = 10
        self.maybe_factor = 1.25

        # initialize our confidence
        self.confidence = 0.95
        self.confidence_interval = [
                2*self.slopes[0],
                0.5*self.slopes[0]]
        self.confidence_interval.sort()
        self.update()

    def slope(self, x1, y1, x2, y2):
        """Calculate slope of given line"""
        if x1 == x2:
            return np.inf
        else:
            return (y2-y1)/(x2-x1)

    def get_best_match(self, lines):
        """Parse the given list of lines for potential matches.

        Args:
            lines: A list of points in [x1,y1,x2,y2] format. 
        """
        # sort given lines into "match" and "not"
        definite = []
        maybe = None
        maybe_slope = np.inf
        for line in lines:
            slope = self.slope(*line)
            if self.confidence_interval[0] <= slope <= self.confidence_interval[1]:
                self.slopes.append(slope)
                definite.append(line)
            else:
                # save best next guess
                if not self.rho:
                    continue
                # check if our error is worse than next best
                if abs(slope-self.rho) > abs(maybe_slope-self.rho):
                    continue
                # construct maybe interval:
                maybe_interval = [
                            self.rho*self.maybe_factor, 
                            self.rho/self.maybe_factor]
                maybe_interval.sort()
                # check if this is within our allowable error (very wide range)
                if maybe_interval[0] <= slope <= maybe_interval[1]:
                    maybe = line
                    maybe_slope = slope

        # add definite lines
        self.matches += definite
        self.matches_per_frame.append(len(definite))
        
        # if we don't have any matches use the next best guess
        if len(definite) == 0:
            # if we didn't get any maybes just return nothing
            if maybe is None:
                return []
            definite.append(maybe)
            self.slopes.append(maybe_slope)
       

        self.update(sum(self.matches_per_frame[-self.n_lanes:]))

        # return our best estimate for the lane, sampled at min/max X of current lines
        definite = np.array(definite)
        X = definite[:,[0,2]].flatten()
        Y = definite[:,[1,3]].flatten()
        fit = np.polynomial.Polynomial.fit(X,Y,1)
        
        x,y = fit.linspace(2)
        return [int(x[0]),int(y[0]),int(x[1]),int(y[1])]


    def update(self, last_n_matches=None):
        """Recalculate our confidence and Line Range with any new data.

        Args:
            last_n_matches: Number of previous matches to use.
        """
        if last_n_matches is None:
            # use all the matches
            last_n_matches = len(self.slopes)

        if len(self.matches_per_frame) < 5:
            print("Note enough matches to update interval.")
            return

        # calculate std deviation of the slope of the last N matches
        np_slopes = np.array(self.slopes)
        m = np.mean(np_slopes)
        se = scipy.stats.sem(np_slopes)
        h = se * scipy.stats.t.ppf((1+self.confidence)/2.0, len(np_slopes)-1)

        ci_old = self.confidence_interval
        self.confidence_interval = [m-h, m+h]
        self.rho = m
        print("CI {}->{}".format(ci_old, self.confidence_interval))
        


