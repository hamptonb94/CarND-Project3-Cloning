import os.path
import random
import numpy as np
from csv import DictReader, DictWriter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

DATA_DIR = 'data'
OUT_DIR  = 'procout'

class LogData:
    def __init__(self, filename = 'driving_log.csv', directory=DATA_DIR):
        self.rows = []
        with open(os.path.join(directory,'driving_log.csv')) as csvfile:
            reader = DictReader(csvfile)
            for row in reader:
                self.rows.append(row)
        print("Driving Log processed. Entries:", len(self.rows))

def findExtremes(logdata):
    minSteer = 1000
    maxSteer = -1000
    for row in logdata:
        steering = float(row['steering'])
        if steering < minSteer:
            minSteer = steering
            caseRight = row
        if steering > maxSteer:
            maxSteer = steering
            caseLeft = row
    return { 'center':logdata[0], 'left':caseLeft, 'right':caseRight }

def findSmallSample(logdata):
    """Find 10 examples (each) of steering left, right, and center"""
    centers = []
    lefts = []
    rights = []
    for row in logdata:
        steering = float(row['steering'])
        if steering < -0.4:
            lefts.append(row)
        if steering > 0.4:
            rights.append(row)
        if steering > -0.0001 and steering < 0.0001 and len(centers) < 10:
            centers.append(row)
    samples = centers
    samples += random.sample(lefts, 10)
    samples += random.sample(rights, 10)
    return samples

def formatImage(logrow, name):
    img = mpimg.imread(os.path.join(DATA_DIR, logrow['center']))
    cropped = img[60:140, 0:320]
    plotImg = plt.imshow(cropped)
    plt.title("Steering = "+logrow['steering']+" CROPPED")
    plt.savefig(OUT_DIR+"/"+name+",cropped.png")

def plotImage(logrow, name):
    img = mpimg.imread(os.path.join(DATA_DIR, logrow['center']))
    plotImg = plt.imshow(img)
    plt.title("Steering = "+logrow['steering'])
    plt.savefig(OUT_DIR+"/"+name+".png")

if __name__ == '__main__':
    logdata = LogData()
    print(logdata.rows[0])

    examples = findExtremes(logdata.rows)
    for name in examples:
        plotImage(examples[name], name)
        formatImage(examples[name], name)
    
    samples = findSmallSample(logdata.rows)
    with open(os.path.join(DATA_DIR,'simple_train.csv'), 'w') as csvfile:
        writer = DictWriter(csvfile, fieldnames=logdata.rows[0].keys())
        writer.writeheader()
        for row in samples:
            writer.writerow(row)
    
    
