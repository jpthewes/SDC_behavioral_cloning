# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/center_lane_driving.png "Center Lane Driving - training"
[image2]: ./images/recovery_from_left.png "Recovery from left lane side - training"
[image3]: ./images/recovery_from_right.png "Recovery from right lane side - training"
[image4]: ./images/tight_right_curve.png "Driving tight right curve - training"

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


My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 73-79) followed by 4 fully connected layers.
The model includes RELU layers after each convolutional layer to introduce nonlinearity. But before the data gets into the convolutional layers the data is normalized in the model using a Keras lambda layer (code line 71). In addition a cropping layer is added to crop the input images to only take a look at the important parts (=the road). 
A summary of the model is plotted each time it is trained and also provided below:
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 80, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 38, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 17, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 7, 37, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 5, 35, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 3, 33, 64)         36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 6336)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               633700    
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 770,619
Trainable params: 770,619
Non-trainable params: 0
_________________________________________________________________

#### 2. Attempts to reduce overfitting in the model

To reduce overfitting I tried to introduce dropout layers but they decreased the driving performance. So therefore I took them out again (lines 75 and 77).
Still the training and validation is splitted and shuffled each epoch (lines 26 and 33)


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 88).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, driving through sharp tunes and swirvy driving.
In addition the images from all three cameras were used and the angles were corrected fo the left and right camera images (lines 39-56)


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the model used by the NVIDIA autonomous research team. I thought this model is appropriate because they work on this topic on a daily basis and tested this architecture well to fit for driving tasks.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a high mean squared error on training and validation data and did not drive well through curves. 
As a next step I then introduced a lambda layer to standardize the data and get a mean around 0. Moreover, I used a Cropping2D layer to crop the input images to only use the relevant parts of the image (=the road). This improved my mean squared error but did not advance the curve driving.
Then I tested introducing Dropout layers, but they somehow made driving even worse, so I let them by the side and tried to move on without them.

What eventually helped was using all 3 camera images (with angle correction) and getting more data from the curvy parts.


At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 73-79) followed by 4 fully connected layers. 
The model includes RELU layers after each convolutional layer to introduce nonlinearity. But before the data gets into the convolutional layers the data is normalized in the model using a Keras lambda layer (code line 71). In addition a cropping layer is added to crop the input images to only take a look at the important parts (=the road). 
A summary of the model is plotted each time it is trained and also provided below:
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 80, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 38, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 17, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 7, 37, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 5, 35, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 3, 33, 64)         36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 6336)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               633700    
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 770,619
Trainable params: 770,619
Non-trainable params: 0
_________________________________________________________________

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering driving around the tight curves at the end of the track:

![alt text][image4]

Furthermore recovering from the sides of the road was recorded:
![alt text][image2]
![alt text][image3]


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 as evidenced by the not decreasing mean squared error after that epoch 4. I used an adam optimizer so that manually training the learning rate wasn't necessary.
