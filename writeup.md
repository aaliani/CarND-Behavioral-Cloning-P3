# **Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3 layers of 5x5, followed by 2 layers of 3x3 filter sizes and depths between 24 and 64 (model.py lines 142 - 184) 

The model includes ELU (Exponential linear unit) activations to introduce nonlinearity and adrss the vanishing gradient problem. The data is normalized in the first layer of model using a Keras lambda layer (code line 142). 

#### 2. Attempts to reduce overfitting in the model

The model contains one dropout layer in order to reduce overfitting (model.py lines 166). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 213-214). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 222).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I took four laps around the track to gather enough data.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to have a relatively simple, but effective model. An architecture too deep with too many parameters would take a lot of time to train and too much processing, hence latency, when running realtime on the simulator. Conversely, too simple a model, like LeNet, would not be effective.

After some research, I decided to go with the architecture inspired from [NVIDIA's 
End to End Learning for Self-Driving Cars (Bojarski et. al. 2016)](https://arxiv.org/abs/1604.07316). This looked perfect for the application and, indeed, proved very effiecient in the end.

My model uses the input shape of 160x320x3. Since the bottom pixels are occupied by the front of the car and top third of the pixels are associated with the sky and acenery, I apply the (60,20) cropping to the input using Keras Crop2D layer.

The cropped image then goes through a series of five convultional layers of three 5x5 filters and then two 3x3 filters. The output of the covultional layers ranges between 24 and 64, in order too keep the parameter count reasonable for realtime performance. Each convolution later is activated using ELU to intrduce non-linearity. The choice of ELU is to address the vanishing gradient problem during training.

In order to avoid overfitting, the output from the convolutional layers is reduced using dropout (of by default 50% with flexibilty provided in the code for the user to change that percentage at runtime when initializing the training). 

The output is then flatened and passed through a series of fully connected layers, which in the end give out the single output for the network, i.e. the steering angle.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. In all epochs, my training loss was very close to the validation loss, which meant that my model was indeed not overfitting.

The final step was to run the simulator to see how well the car was driving around track one. I could see that the car was staying very close to the right side of the road. I made left turns fine but completely fell off on the right turns. That made sense beacuase the traning data was heavily biased with left turns throughout the track. There was only one right turn in the track so it made sense that the model could not learn it well. 

In order to combat that, I decided to flip randomly between 30% to 70% of the samples and invert their respective steering angles in each batch so that the model learns both left and right turns. And that did the trick.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 142-184) consisted of a convolution neural network with the following layers and layer sizes:

Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 80, 280, 3)    0           lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 38, 138, 24)   1824        cropping2d_1[0][0]               
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 17, 67, 36)    21636       convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 7, 32, 48)     43248       convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 5, 30, 64)     27712       convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 3, 28, 64)     36928       convolution2d_4[0][0]            
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 3, 28, 64)     0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 5376)          0           dropout_1[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           537700      flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]                    
====================================================================================================
Total params: 674,619
Trainable params: 674,619
Non-trainable params: 0


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded four laps on track one using center lane driving. 

I used mouse to record the data as it gives smoother results.

In order for my network to learn to sharply turn if it is too close to the edges, I recorded the data at maximum speed with lots of sharp turns.

To augment the data sat, I also flipped between 30% and 70% of the images and angles in each batch.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by the fact that my training and validation loss stopped significantly decreasing after that. I used an adam optimizer so that manually training the learning rate wasn't necessary.
