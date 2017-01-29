import os.path
import argparse
import random
import numpy as np
import cv2
import matplotlib.image as mpimg
from csv import DictReader, DictWriter
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, MaxPooling2D, Convolution2D
from keras.regularizers import l2

N_EPOCHS = 5
KEEP     = 0.2
R_VALID  = 0.15
CONV_INIT='glorot_uniform'

# pre-process data:
N_ROWS = 160-80
N_COLS = 320
IMG_SHAPE = (N_ROWS, N_COLS, 3)


DATA_DIR = 'data'
class LogData:
    """Class to load the log data and split into training/validation"""
    def __init__(self, filename = 'driving_log.csv', directory=DATA_DIR):
        self.rows = []
        with open(os.path.join(directory, filename)) as csvfile:
            reader = DictReader(csvfile)
            for row in reader:
                self.rows.append(row)
        print("Driving Log processed. Entries:", len(self.rows))
        random.shuffle(self.rows)
        split = int((1.0-R_VALID)*len(self.rows))
        self.trainRows = self.rows[:split]
        self.validRows = self.rows[split:]
        print("Training/Validation sizes:", len(self.trainRows), len(self.validRows))


def driveModelSimple():
    """Super simple NN model to test process flow"""
    
    model = Sequential()
    # Normalize each pixel
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=IMG_SHAPE, output_shape=IMG_SHAPE))
    
    # 1st Convolution
    model.add(Convolution2D(24, 3, 3, subsample=(2,2), name="Conv2D_1", input_shape=IMG_SHAPE))
    model.add(ELU())
    
    # Flatten for connected layers
    model.add(Flatten(name="Flatten"))
    
    # Fully connect to output
    model.add(Dense(1, name="Output"))
    
    model.compile(loss="mse", optimizer="adam")
    return model

def driveModelNvidia():
    """NVIDIA Steering model"""
    
    model = Sequential()
    # Normalize each pixel
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=IMG_SHAPE, output_shape=IMG_SHAPE))
    
    # NVIDIA Steering model
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid", init=CONV_INIT, W_regularizer=l2(0.01)))
    model.add(ELU())
    model.add(Dropout(KEEP))

    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid", init=CONV_INIT))
    model.add(ELU())
    model.add(Dropout(KEEP))
    
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid", init=CONV_INIT))
    model.add(ELU())
    model.add(Dropout(KEEP))

    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init=CONV_INIT))
    model.add(ELU())
    model.add(Dropout(KEEP))

    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init=CONV_INIT))
    model.add(ELU())
    model.add(Dropout(KEEP))
    
    model.add(Flatten())

    model.add(Dense(100))
    model.add(ELU())
    model.add(Dropout(0.2))
    
    model.add(Dense(50))
    model.add(ELU())
    model.add(Dropout(0.2))
    
    model.add(Dense(10))
    model.add(ELU())
    
    model.add(Dense(1))
    
    model.compile(loss="mse", optimizer="adam")
    return model

def driveModel(modelName):
    if modelName == 'simple':
        return driveModelSimple()
    if modelName == 'nvidia':
        return driveModelNvidia()
    raise "Unknown model:"+modelName


def generateTrainingBatch(logdata, batch_size):
    """For training we will use all camera angles, and do augmentation"""
    while logdata:
        batch_x = []
        batch_y = []
        for row in logdata.trainRows:
            # we will choose between the 3 cameras and get the image
            camera = random.choice(['left','center','right'])
            image  = mpimg.imread(os.path.join(DATA_DIR, row[camera].strip()))
            image  = image[60:140, 0:320]  # crop top and bottom
            
            steering = float(row['steering'])
            #adjust steering for left/right camera angles
            if camera == 'left' : steering += 0.2
            if camera == 'right': steering -= 0.2
                        
            batch_x.append(np.reshape(image, (1, N_ROWS, N_COLS, 3)))
            batch_y.append(np.array([[steering]]))
            
            # to augment data with left/right steering we will flip images
            # that have non-zero steering which will double those samples
            if steering < -0.01 or steering > 0.01 and len(batch_x) < batch_size:
                image = cv2.flip(image, 1)
                steering *= -1
                batch_x.append(np.reshape(image, (1, N_ROWS, N_COLS, 3)))
                batch_y.append(np.array([[steering]]))
            
            # yield an entire batch at once
            if len(batch_x) == batch_size:
                batch_x, batch_y, = shuffle(batch_x, batch_y, random_state=21)
                yield (np.vstack(batch_x), np.vstack(batch_y))
                batch_x = []
                batch_y = []

def generateValidationBatch(logdata, batch_size):
    """For validation we just return a batch of center images"""
    while logdata:
        batch_x = []
        batch_y = []
        for row in logdata.validRows:
            camera = 'center'
            image  = mpimg.imread(os.path.join(DATA_DIR, row[camera].strip()))
            image  = image[60:140, 0:320]  # crop top and bottom
            steering = float(row['steering'])
            batch_x.append(np.reshape(image, (1, N_ROWS, N_COLS, 3)))
            batch_y.append(np.array([[steering]]))
            # yield an entire batch at once
            if len(batch_x) == batch_size:
                batch_x, batch_y, = shuffle(batch_x, batch_y, random_state=21)
                yield (np.vstack(batch_x), np.vstack(batch_y))
                batch_x = []
                batch_y = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('modelName', type=str, help='Name of NN model to train.')
    parser.add_argument('batchSize', type=int, help='Number of images per batch.', nargs='?', default= 100)
    parser.add_argument('numTrain',  type=int, help='Number of images per epoch.', nargs='?', default=8000)
    args = parser.parse_args()
    
    # set up input log data
    logName = 'simple_train.csv' if args.modelName == 'simple' else 'driving_log.csv'
    logdata = LogData(filename = logName)
    
    model = driveModel(args.modelName)
    model.summary()
    
    generatorTrain = generateTrainingBatch(  logdata, args.batchSize)
    generatorValid = generateValidationBatch(logdata, args.batchSize)
    
    history = model.fit_generator(
                generator = generatorTrain,
                samples_per_epoch = args.numTrain,
                nb_epoch = N_EPOCHS,
                verbose = 2,
                validation_data = generatorValid,
                nb_val_samples  = args.numTrain*0.2  )
    
    model.save_weights(args.modelName+".h5")
    open(args.modelName+".json", "w").write(model.to_json(indent=4))


# For simple model use batchSize = 25, numTrain = 400


