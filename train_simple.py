import argparse
import numpy as np
import cv2
from preprocess import *
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, MaxPooling2D, Convolution2D
from keras.regularizers import l2

N_EPOCHS = 5
KEEP     = 0.2
CONV_INIT='glorot_uniform'

# pre-process data:
N_ROWS = 160-80
N_COLS = 320
IMG_SHAPE = (N_ROWS, N_COLS, 3)

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


def generateBatch(logdata, batch_size):
    while logdata:
        batch_x = []
        batch_y = []
        for row in logdata.rows:
            # there are so many images with no steering angle
            # so we will skip 1/2 of them
            if random.choice(['Skip', 'Keep']) == 'Skip':
                continue
            
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
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('modelName', type=str, help='Name of NN model to train.')
    parser.add_argument('batchSize', type=int, help='Number of images per batch.', nargs='?', default=128)
    parser.add_argument('numTrain',  type=int, help='Number of images per epoch.', nargs='?', default=4000)
    args = parser.parse_args()
    
    # set up input log data
    logName = 'simple_train.csv' if args.modelName == 'simple' else 'driving_log.csv'
    logdata = LogData(filename = logName)
    
    model = driveModel(args.modelName)
    model.summary()
    
    generatorTrain = generateBatch(logdata, args.batchSize)
    generatorValid = generateBatch(logdata, args.batchSize)
    
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


