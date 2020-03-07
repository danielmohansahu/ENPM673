# ENPM673 Project #2: Lane Detection

This project attempts to implement lane detection on the two given data sets.

### Dependencies:

We use Python3 and the following non-standard modules:
 - matplotlib
 - yaml
 - numpy
 - scipy

All can be installed via:

```bash
pip install numpy yaml matplotlib scipy
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

Each data set has its own custom script. To run dataset #1, call:  
```bash
./process_data1.py -i PATH_TO_IMAGES
```

This will find all the images related to dataset #1, combine them into a video, process that video, and output the video with lanes detected where the script is ran.

Running dataset #2 is similar, i.e.:  
```bash
./process_data2.py -i PATH_TO_IMAGES
```

Frame-by-frame debugging can be enabled via the `-d True` flag. This will show every stage of processing for each frame. To continue to the next frame hit any key. To stop showing images entirely hit `q`.

Note that, although you can optionally supply the intrinsics calibration YAML we recommend using the local one. This is because the YAMLs provided in the problem definition are not properly formatted and cannot be parsed.

