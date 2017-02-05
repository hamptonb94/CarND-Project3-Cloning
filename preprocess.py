import os.path
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from csv import DictReader, DictWriter

DATA_DIR = 'data'
OUT_DIR  = 'procout'

class LogData:
    def __init__(self, filename = 'driving_log.csv', directory=DATA_DIR):
        self.rows = []
        with open(os.path.join(directory, filename)) as csvfile:
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

def findSmallSample(logdata, num_samples = 10):
    """Find 10 examples (each) of steering left, right, and center"""
    centers = []
    lefts = []
    rights = []
    for row in logdata:
        steering = float(row['steering'])
        if steering < -0.2:
            lefts.append(row)
        if steering > 0.2:
            rights.append(row)
        if steering > -0.0001 and steering < 0.0001 and len(centers) < num_samples:
            centers.append(row)
    samples = centers
    samples += random.sample(lefts, num_samples)
    samples += random.sample(rights, num_samples)
    return samples

def formatImage(logrow, name):
    img = mpimg.imread(os.path.join(DATA_DIR, logrow['center']))
    cropped = img[60:140, 0:320]
    plotImg = plt.imshow(cropped)
    plt.title("Steering = "+logrow['steering']+" CROPPED")
    plt.savefig(OUT_DIR+"/"+name+",cropped.png")
    mpimg.imsave(OUT_DIR+"/"+name+",cropped,raw.png", cropped)
    small = cv2.resize(cropped, (200,66))
    plotImg = plt.imshow(small)
    plt.title("Steering = "+logrow['steering']+" RESIZED")
    plt.savefig(OUT_DIR+"/"+name+",cropped,rs.png")
    i = 0
    for brightness in np.arange(0.3,1.0,0.1):
        i += 1
        imageHSV        = cv2.cvtColor(small,cv2.COLOR_RGB2HSV)
        imageHSV[:,:,2] = imageHSV[:,:,2]*brightness
        image2          = cv2.cvtColor(imageHSV,cv2.COLOR_HSV2RGB)
        plotImg = plt.imshow(image2)
        plt.title("Steering = "+logrow['steering']+" brightness={0:.3f}".format(brightness))
        plt.savefig(OUT_DIR+"/"+name+",cropped,rs,b"+repr(i)+".png")
        mpimg.imsave(OUT_DIR+"/"+name+",cropped,rs,b"+repr(i)+",raw.png", image2)

def formatImage2(logrow, name):
    image = mpimg.imread(os.path.join(DATA_DIR, logrow['center']))
    image = cv2.resize(image, (200,100))
    plotImg = plt.imshow(image)
    plt.title("Steering = "+logrow['steering']+" RESIZED")
    plt.savefig(OUT_DIR+"/"+name+",sized1.png")
    cropped = image[28:94, 0:200]
    plotImg = plt.imshow(cropped)
    plt.title("Steering = "+logrow['steering']+" CROPPED")
    plt.savefig(OUT_DIR+"/"+name+",cropped2.png") 

def plotImage(logrow, name):
    img = mpimg.imread(os.path.join(DATA_DIR, logrow['center']))
    plotImg = plt.imshow(img)
    plt.title("Steering = "+logrow['steering'])
    plt.savefig(OUT_DIR+"/"+name+".png")
    if name == 'left':
        image2 = cv2.flip(img, 1)
        plotImg = plt.imshow(image2)
        newAngle = -1 * float(logrow['steering'])
        plt.title("Steering = {} (Flipped image)".format(newAngle))
        plt.savefig(OUT_DIR+"/"+name+",flip.png")

def plotHistogram(logdata):
    angles = []
    for row in logdata.rows:
        angles.append(float(row['steering']))
    n, bins, patches = plt.hist(angles, 50)
    plt.title("Steering Angle Counts")
    plt.savefig(OUT_DIR+"/hist.png")

if __name__ == '__main__':
    logdata = LogData()
    
    plotHistogram(logdata)
    
    examples = findExtremes(logdata.rows)
    for name in examples:
        plotImage(examples[name], name)
        formatImage(examples[name], name)
        formatImage2(examples[name], name)
    
    samples = findSmallSample(logdata.rows, num_samples = 100)
    with open(os.path.join(DATA_DIR,'simple_train.csv'), 'w') as csvfile:
        writer = DictWriter(csvfile, fieldnames=logdata.rows[0].keys())
        writer.writeheader()
        for row in samples:
            writer.writerow(row)
    
    
