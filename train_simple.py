import numpy as np
import cv2
from preprocess import *
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, MaxPooling2D, Convolution2D

LOG_FILE = 'simple_train.csv'
MODEL_NAME = 'model_s'
N_EPOCHS = 5
BATCH_SIZE=25
N_SAMP_TRAIN = 400
N_SAMP_VALID = N_SAMP_TRAIN * 0.2

# pre-process data:
N_ROWS = 160-80
N_COLS = 320
IMG_SHAPE = (N_ROWS, N_COLS, 3)

def driveModel():
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


def generateBatch(logdata, batch_size=128):
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
    
    logdata = LogData(filename = LOG_FILE)
    
    model = driveModel()
    model.summary()
    
    generatorTrain = generateBatch(logdata, BATCH_SIZE)
    generatorValid = generateBatch(logdata, BATCH_SIZE)
    
    history = model.fit_generator(
                generator = generatorTrain,
                samples_per_epoch = N_SAMP_TRAIN,
                nb_epoch = N_EPOCHS,
                verbose = 2,
                validation_data = generatorValid,
                nb_val_samples  = N_SAMP_VALID  )
    
    model.save_weights(MODEL_NAME+".h5")
    open(MODEL_NAME+".json", "w").write(model.to_json(indent=4))
