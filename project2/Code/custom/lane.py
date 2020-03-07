"""Lane Class; used for tracking/probablistic estimate of lanes.
"""
import cv2
import numpy as np
import scipy.stats

# TUNING PARAMS
MINIMUM_FRAMES=5
FRAMES_TO_UPDATE=5
NEW_SLOPE_MARGIN=15
MIN_ANGLE=5
MAX_ANGLE=85
CONFIDENCE=0.95
MAX_RESIDUAL=100000

class Lane:
    """This class maintains a list of lines thought to be in the same lane.

    All analysis is performed on the slope of the lane. Since this changes
    over time we keep track of our confidence on the last few detections.
    """
    def __init__(self, seed):
        # initialize our list of matched lanes with a single expected lane.
        self.matches = [seed]
        self.slopes = [self.slope(*seed)]
        self.rho = None
        
        # keep track of how many matches we have per frame
        self.matches_per_frame = [1]

        # various tuning parameters
        self.max_residual = MAX_RESIDUAL
        self.min_frames = MINIMUM_FRAMES
        self.n_lanes = FRAMES_TO_UPDATE
        self.maybe_degrees = NEW_SLOPE_MARGIN*np.math.pi/180
        self.slope_interval = [
                MIN_ANGLE*np.math.pi/180, 
                MAX_ANGLE*np.math.pi/180
        ]

        # initialize our confidence
        self.confidence = CONFIDENCE
        self.confidence_interval = [
                2*self.slopes[0],
                0.5*self.slopes[0]]
        self.confidence_interval.sort()

        # initial update
        self.calc_confidence()

    def update(self, lines):
        """Parse the given list of lines for potential matches.

        Args:
            lines: A list of points in [x1,y1,x2,y2] format. 
        """
        
        definite = []
        maybe = None
        maybe_slope = np.inf
        
        # sort given lines into "match" and "not"
        for line in lines:
            # calculate slope (as angle):
            slope = self.slope(*line)

            # filter out vertical / horizontal slopes
            if not self.filter_slope(slope):
                continue
            
            # check if it's a good line
            if self.confidence_interval[0] <= slope <= self.confidence_interval[1]:
                self.slopes.append(slope)
                definite.append(line)
            else:
                # keep track of the closest "bad" line
                # check if our error is worse than next best
                if not self.rho or (abs(slope-self.rho) > abs(maybe_slope-self.rho)):
                    continue

                # construct maybe interval:
                maybe_interval = [
                        self.rho + self.maybe_degrees, 
                        self.rho - self.maybe_degrees]
                maybe_interval.sort()
                
                # check if this is within our allowable error (very wide range)
                if maybe_interval[0] <= slope <= maybe_interval[1]:
                    maybe = line
                    maybe_slope = slope
        
        # if we don't have any matches use the next best guess
        if len(definite) == 0:
            # if we didn't get any maybes just return nothing
            if maybe is None:
                return 
            definite.append(maybe)
            self.slopes.append(maybe_slope)
        
        # add matches to our list
        self.matches += definite
        self.matches_per_frame.append(len(definite))
        
        # update with matches from the last N frames
        matches_to_update = sum(self.matches_per_frame[-self.n_lanes:])
        self.calc_confidence(matches_to_update)

    def predict(self):
        """Return our best estimate for the lane.

        This fits a second degree polynomial to the last few lane matches.
        """
        # get the last N lane estimates:
        last_estimates = np.array(self.matches[-sum(self.matches_per_frame[-self.n_lanes:]):])

        # fit a line to each of these and sample in our X domain
        X = []
        Y = []
        for x1,y1,x2,y2 in last_estimates:
            # ignore vertical lines
            if x1==x2:
                continue
            line_fit = np.polynomial.Polynomial.fit([x1,x2],[y1,y2],1)
            x,y = line_fit.linspace()
            X += x.tolist()
            Y += y.tolist()
        
        # try to fit a second order polynomial to our lines 
        fit,res = np.polynomial.Polynomial.fit(X,Y,2,full=True)
        if res[0][0] > self.max_residual:
            return None
        return fit

    def plot(self, frame, fit, thickness=2):
        """Plot the given polynomial on the given frame.
        """
        if not fit:
            return frame
        X,Y = fit.linspace()
        X = np.array(X,dtype=np.int32)
        Y = np.array(Y,dtype=np.int32)
        curve = np.column_stack((X,Y))
        cv2.polylines(frame, [curve], False, color=(0,0,255),thickness=thickness)
        return frame

    def calc_confidence(self, last_n_matches=None):
        """Recalculate our confidence and Line Range with any new data.

        Args:
            last_n_matches: Number of previous matches to use.
        """
        if last_n_matches is None:
            # use all the matches
            last_n_matches = len(self.slopes)

        # only start processing once we've gotten a few frames
        if len(self.matches_per_frame) < self.min_frames:
            return
        
        # calculate confidence interval of the slope of the last N matches
        np_slopes = np.array(self.slopes[-last_n_matches:])
        m = np.mean(np_slopes)
        se = scipy.stats.sem(np_slopes)
        h = se * scipy.stats.t.ppf((1+self.confidence)/2.0, len(np_slopes)-1)

        # update class variables
        self.confidence_interval = [m-h, m+h]
        self.rho = m
        
    def slope(self, x1, y1, x2, y2):
        """Calculate normalized (0,pi/2) slope of given line"""
        if x1 == x2:
            slope = np.inf
        else:
            slope = (y2-y1)/(x2-x1)

        return np.math.atan(slope)

    def filter_slope(self,slope):
        """Apply heuristics to slopes (e.g. no vertical/horizontal lines.
        """
        if self.slope_interval[0] <= abs(slope) <= self.slope_interval[1]:
            return True
        return False

