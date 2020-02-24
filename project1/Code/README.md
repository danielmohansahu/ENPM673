## ENPM673 Project #1

This repository contains the necessary code to detect and track a fiducial tag. A given input video will be processed such that any tags detected will be replaced with a given reference image and a 3D cube.

### Usage:

Running the core processing code is straightforward from the command line. Assuming that you have the files `Tag0.mp4`, `ref_marker.png`, and `Lena.png` in your working directory:

```bash
./solve.py --video Tag0.mp4 --tag_file ref_marker.png --reference_file Lena.png
```

Those input files also happen to be the defaults. So to run the same process against a different video file just call:

```bash
./solve.py --video Tag2.mp4
```

With the default verbosity settings (optionally set via `--verbose N` with `0 <= N <=2`) you should see something similar to the following output:

```bash
Found ids [0] in frame #1/554 in 0.160s (0.245s total)
Found ids [0] in frame #2/554 in 0.176s (0.443s total)
Found ids [0] in frame #3/554 in 0.159s (0.609s total)
Found ids [0] in frame #4/554 in 0.167s (0.782s total)
...
```

After all the frames are processed a video file will be created in your current directory name `processed_{INPUT_VIDEO_NAME}.mp4` containing the output video.

### Prerequisites:

This code was written and tested on Ubuntu 18.04 using Python3. The only non-standard modules used are OpenCV and Numpy.

### Notes

This code is theoretically runnable against any given set of input images. However, a number of assumptions were made about detections (specifically number of fiducial sides and method of identification) that will make this code difficult to run against arbitrary data sets.