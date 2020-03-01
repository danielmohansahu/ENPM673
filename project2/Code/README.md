# ENPM673 Project #2: Lane Detection

This project attempts to implement lane detection on the two given data sets.

### Dependencies:

We use Python3 and the following non-standard modules:
 - matplotlib
 - yaml
 - numpy

All can be installed via:

```bash
pip install numpy yaml matplotlib
``` 

### Usage:

##### Part #1

Part #1 can be ran via the following script:

```bash
./pre_process.py
```

TODO:
 - really just play with this. the pre_processing isn't doing very well, but the structure is in place to try lots of different algorithms to make it better. This might apply well to Part #2 as well.

##### Part #2
We're only running dataset #1 so far. Currently we assume the following:

 - Data is stored relative to the codebase in the ../Data/ directory.


The first thing we need to do is convert our data set to a video (as they're given as isolated images). We do this via the following:
```bash
./make_video.py "../Data/data_1/data/*.png" -o ../Data/data_1/data_1.mp4
```

Then we can run the core script:
```bash
./detect_lanes.py
```

@TODO:
 - I've gotten as far as to try to do line fitting to the warped image (i.e. approach #2 specified in the problem description). 
 - the solution I've implemented isn't very robust at all; there are lots of parts which are problematic
 - The bulk of the work remaining (besides improving the robustness) is to determine which detected lines are actually lanes and to transform them back into the original image coordinates. Then we can do all the fancy stuff (e.g. turn prediction)
 - See lines 40-68 of `detect_lanes.py` for a top level view of everything I'm currently doing.
