
import numpy as np
from preprocess import *
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, MaxPooling2D, Convolution2D



N_ROWS = 160-80
N_COLS = 320
IMG_SHAPE = (N_ROWS, N_COLS, 3)

def simpleModel():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=IMG_SHAPE, output_shape=IMG_SHAPE))
    model.add(Convolution2D(24, 3, 3, subsample=(2,2), name="Conv2D_1", input_shape=IMG_SHAPE))
    model.add(ELU())
    model.add(Flatten(name="Flatten"))
    model.add(Dense(1, name="Output"))
    model.compile(loss="mse", optimizer="adam")
    return model

def generateBatch(logdata, batch_size=128):
    while logdata:
        batch_x = []
        batch_y = []
        for row in logdata.rows:
            image = mpimg.imread(os.path.join(DATA_DIR, row['center']))
            image = image[60:140, 0:320]  # crop top and bottom
            steering = float(row['steering'])
            
            batch_x.append(np.reshape(image, (1, N_ROWS, N_COLS, 3)))
            batch_y.append(np.array([[steering]]))
            
            if len(batch_x) == batch_size:
                batch_x, batch_y, = shuffle(batch_x, batch_y, random_state=21)
                yield (np.vstack(batch_x), np.vstack(batch_y))
                batch_x = []
                batch_y = []


INPUT_SHAPE = [160, 320, 3]
SAMPLES_PER_EPOCH = 150
VALID_SAMPLES_PER_EPOCH = 50
EPOCHS = 5

if __name__ == '__main__':
    model = simpleModel()
    model.summary()
    #plot(model, to_file="model.png", show_shapes=True)
    
    logdata = LogData('simple_train.csv')
    
    generatorTrain = generateBatch(logdata, 30)
    generatorValid = generateBatch(logdata, 30)
    
    history = model.fit_generator( generatorTrain,
                SAMPLES_PER_EPOCH,
                EPOCHS,
                validation_data=generatorValid, verbose=2,
                nb_val_samples=VALID_SAMPLES_PER_EPOCH)
    
    model.save_weights("model_s.h5")
    open("model_s.json", "w").write(model.to_json(indent=4))
