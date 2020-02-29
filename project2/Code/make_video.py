#!/usr/bin/env python3

import glob
import argparse
import code
from custom.file_utils import pics2video 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("pictures", help="regex expression to match image files.")
    parser.add_argument("-o","--output",default="video.mp4",help="Name of the output video.")
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    
    # get all pictures
    image_files = glob.glob(args.pictures)

    # convert to video
    pics2video(image_files, args.output)

    # code.interact(local=locals())
