import pandas as pd 
import numpy as np 
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle 
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Convolution2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.layers import Cropping2D
import cv2
import argparse
import os


# suppress tensorflow warnings
# enables using larger batch size with cleaner console output
# otherwise after each batch there is a warning for GPU memory use >10%
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def generator(args, samples_X, samples_y, batch_size=32):

    """
    A generator function
    """

    # make sure the length of X and y sample sets is same
    assert len(samples_X) == len(samples_y)

    # number of samples
    num_samples = len(samples_y)

    # indefinite loop to keep spitting out batches for as long as needed
    while 1:

        # shuffle samples 
        samples_X, samples_y = shuffle(samples_X, samples_y)

        # create batches of batch_size until the whole sample set is run through
        for offset in range(0, num_samples, batch_size):

            # batch of batch_size for this iteration of x and y samples
            batch_samples_X = samples_X[offset:offset+batch_size]
            batch_samples_y = samples_y[offset:offset+batch_size]

            # empty arrays to store the images X and steering angles y of this batch
            images = []
            angles = []
            
            # load images for this batch from drive based on filepaths in batch_samples_X
            # and store it into the images array
            for batch_sample in batch_samples_X:
                name = args.data_dir + '/IMG/' + batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)[...,::-1]
                images.append(center_image)
            
            # store the steering angles from batch_sample_y into angles array
            for batch_sample in batch_samples_y:
                center_angle = float(batch_sample)
                angles.append(center_angle)

            # convert images and angles into numpy arrays
            X_sample = np.array(images)
            y_sample = np.array(angles)

            # set any nan values to 0
            # Note: Perhaps unneccessary but my model loss would always converge to nan quickly
            # after starting the training until this step. I changed other things too so maybe 
            # that wasn't because of this, but it is nevertheless a safe and stable approach.   
            X_sample[np.isnan(X_sample)] = 0
            y_sample[np.isnan(y_sample)] = 0

            ## Pipeline to create flipped images so that the network also learns the right turns 
            ## which were very few in the training data given the track of the lap

            # randomly select the amount of flipped images to be used for this batch, 
            # between 30%  to 70% of the batch_size 
            n_flip = np.random.randint(int(batch_size * 0.3), int(batch_size * 0.7))

            # amount of original images to keep in the batch 
            n_orig = batch_size - n_flip

            # flip all the images in the batch and invert the corresponding steering angles
            X_flip = np.array([np.fliplr(img) for img in X_sample])
            y_flip = -y_sample

            # shuffle both the original batch and flipped batch
            X_flip, y_flip = shuffle(X_flip, y_flip)
            X_sample, y_sample = shuffle(X_sample, y_sample)

            # select only the randomly allocated amounts of the original and flipped samples, 
            # respectively, from the batches and concatenate them into single output for the batch 
            X_out = np.concatenate((X_sample[:n_orig], X_flip[:n_flip]))
            y_out = np.concatenate((y_sample[:n_orig], y_flip[:n_flip]))

            # shuffle this batch and yield it
            yield shuffle(X_out, y_out)

def load_data(args):
    """
    Adopted from: https://github.com/llSourcell/How_to_simulate_a_self_driving_car/blob/master/model.py
    """

    """
    Load training data and split it into training and validation set
    """
    
    # reads CSV file into a single dataframe variable
    data_df = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

    # we select rows and columns by their names
    # we'll store the camera images as our input data
    # we only use the center image for this project. left and right could ofc be used for better results
    X = data_df[['center']].values
    
    # steering commands as our output data
    y = data_df['steering'].values

    # now we can split the data into a training (80), testing(20), and validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)

    return X_train, X_valid, y_train, y_valid


def build_model(args):

    """
    Adopted from: https://github.com/llSourcell/How_to_simulate_a_self_driving_car/blob/master/model.py
    """

    """
    NVIDIA model used

    the convolution layers are meant to handle feature engineering
    the fully connected layer for predicting the steering angle.
    dropout avoids overfitting
    ELU(Exponential linear unit) function takes care of the Vanishing gradient problem. 
    """
  
    # Initialize the sequential model
    model = Sequential()

    # Image normalization to avoid saturation and make gradients work better.
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

    # Cropping: (up, down, left, right) => (60, 20, 20, 20)
    model.add(Cropping2D(cropping=((60,20), (20,20)), input_shape=(160,320,3)))

    # Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU   
    model.add(Convolution2D(24, 5, 5, activation='elu', subsample=(2, 2)))

    # Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
    model.add(Convolution2D(36, 5, 5, activation='elu', subsample=(2, 2)))

    # Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
    model.add(Convolution2D(48, 5, 5, activation='elu', subsample=(2, 2)))

    # Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    model.add(Convolution2D(64, 3, 3, activation='elu'))

     # Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    model.add(Convolution2D(64, 3, 3, activation='elu'))

    # Drop out (0.5) to avoid overfitting 
    model.add(Dropout(args.keep_prob))

    # Flatten output
    model.add(Flatten())

    # Fully connected: neurons: 100, activation: ELU
    model.add(Dense(100, activation='elu'))

    # Fully connected: neurons: 50, activation: ELU
    model.add(Dense(50, activation='elu'))

    # Fully connected: neurons: 10, activation: ELU
    model.add(Dense(10, activation='elu'))

    # Fully connected: neurons: 1 (output) i.e. the steering angle
    model.add(Dense(1))

    # print model summary
    model.summary()

    return model


def train_model(model, args, X_train, X_valid, y_train, y_valid):

    """
    Adopted from: https://github.com/llSourcell/How_to_simulate_a_self_driving_car/blob/master/model.py
    """


    """
    Train the model
    """
    #Saves the model after every epoch.
    #quantity to monitor, verbosity i.e logging mode (0 or 1), 
    #if save_best_only is true the latest best model according to the quantity monitored will not be overwritten.
    #mode: one of {auto, min, max}. If save_best_only=True, the decision to overwrite the current save file is
    # made based on either the maximization or the minimization of the monitored quantity. For val_acc, 
    #this should be max, for val_loss this should be min, etc. In auto mode, the direction is automatically
    # inferred from the name of the monitored quantity.
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')

    # generators for train and validation sets
    train_generator = generator(args, X_train, y_train, batch_size=args.batch_size)
    validation_generator = generator(args, X_valid, y_valid, batch_size=args.batch_size)

    #calculate the difference between expected steering angle and actual steering angle
    #square the difference
    #add up all those differences for as many data points as we have
    #divide by the number of them
    #that value is our mean squared error! this is what we want to minimize via
    #gradient descent
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))

    #Fits the model on data generated batch-by-batch by a Python generator.

    #The generator is run in parallel to the model, for efficiency. 
    #For instance, this allows you to do real-time data augmentation on images on CPU in 
    #parallel to training your model on GPU.
    #so we reshape our data into their appropriate batches and train our model simulatenously
    model.fit_generator(train_generator,
                        args.samples_per_epoch,
                        args.nb_epoch,
                        validation_data=validation_generator, 
                        nb_val_samples=len(X_valid),
                        callbacks=[checkpoint],
                        verbose=1)

def main():

    """
    Adopted from: https://github.com/llSourcell/How_to_simulate_a_self_driving_car/blob/master/model.py
    """

    """
    Load train/validation data set and train the model
    """
    
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data')
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=10)
    parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=10000)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=32)
    parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=str,   default='true')
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=0.001)
    args = parser.parse_args()

    #print parameters
    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    #load data
    X_train, X_valid, y_train, y_valid = load_data(args)
    
    #build model
    model = build_model(args)
    
    #train model on data, it saves as model.h5 
    train_model(model, args, X_train, X_valid, y_train, y_valid)


if __name__ == '__main__':
    main()

