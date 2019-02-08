# **Traffic Sign Recognition** 

## Writeup

**By Abeer Ghander**


### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report/histogram_original.PNG "histogram_original"
[image11]: ./report/histogram_updated.PNG "histogram_updated"
[image2]: ./report/Sample_images.PNG "Sample images"
[image3]: ./report/grayscale.PNG "Grayscaling"
[image31]: ./report/translated.PNG "Translated"
[image311]: ./report/translated_.PNG "Translated_"
[image32]: ./report/warped.PNG "Warped"
[image321]: ./report/warped_.PNG "Warped"
[image33]: ./report/scaled.PNG "Scaling"
[image331]: ./report/scaled_.PNG "Scaling"
[image34]: ./report/brightness.PNG "Brightness"
[image341]: ./report/brightness_.PNG "Brightness"
[image41]: ./my_signs/pic1.png "Traffic Sign 1"
[image42]: ./my_signs/pic2.png "Traffic Sign 2"
[image43]: ./my_signs/pic3.png "Traffic Sign 3"
[image44]: ./my_signs/pic4.png "Traffic Sign 4"
[image45]: ./my_signs/pic5.png "Traffic Sign 5"
[image46]: ./my_signs/pic6.png "Traffic Sign 6"
[image47]: ./my_signs/pic9.png "Traffic Sign 7"
[image48]: ./my_signs/pic10.png "Traffic Sign 8"
[image5]: ./report/softmax.PNG "Softmax"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

The writeup.. You're reading it! The code is in the workspace here.. [project code](./Traffic_Sign_Classifier.ipynb)
The output of the code as html is here.. [HTML notebook output](./Traffic_Sign_Classifier.html)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I calculated the summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

Sample images from the dataset:
![alt text][image2]

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed in the training dataset.

* Minimum number of images per class/label in training dataset is 180
* Maximum number of images per class/label in training dataset is 2010

![alt text][image1]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it is more efficient to work on grayscaled images. Then we do not need to work on 3 axis (R, G, B) but one axis (Grayscale)

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image3]

As a last step, I normalized the image data because the average for each of the dataset images was not around the 0. So, to treat all images in the same scale, they were normalized using the equation: 'X_data = (X_data - 128.0) / 128'
The normalization step changed the dataset mean as follows:

| Dataset           | Before Normalization | After Normalization |
|:-----------------:|:--------------------:|:-------------------:|
|Training dataset   |82.677589037          |-0.354081335648      |
|Testing dataset    |82.1484603612         |-0.358215153428      |
|Validation dataset |83.5564273756         |-0.347215411128      |

I decided to generate additional data because the current distribution of examples per label/class is not enough for proper training for the network. To avoid underfitting of the network, enough data should be generated. I implemented code to increase dataset data to be at least 800 examples per class (only if the examples are below that number per class)

To add more data to the the data set, I used computer vision techniques to change the pictures randomly and regenerate new picture from existing data. The added functions are: 
* Translate 
![alt text][image31]
![alt text][image311]
* Scale
![alt text][image33]
![alt text][image331]
* Brightness 
![alt text][image34]
![alt text][image341]
* Warp
![alt text][image32]
![alt text][image321]


The difference between the original data set and the augmented data set is the following:
* Original dataset is recorded from the street directly
* Augmented dataset is taken from the original dataset, and random small changes are applied to those images to simulate the same signs from different perspectives/brightness/conditions...

Generation of augmented images is done by creating new image
```
new_img = random_translate(random_scaling(random_warp(random_brightness(new_img))))
```

After adjusting the dataset with the augmented data, the histogram of the examples per class became as follows:
![alt text][image11]

Now we have enough data in our training dataset to start the neural network...

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers (implemented in the function `LeNet2()`):


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image 						| 
| Convolution 5x5 (L1) 	| 1x1 stride, valid padding, outputs 28x28x6 	| 
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x16 	|
| Convolution 5x5 (L2)  | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x16 	|
| Convolution 5x5 (L3)  | 1x1 stride, valid padding, outputs 1x1x400	|
| RELU					|												|
| Flatten L2			| output 400									|
| Concatenate L2 + L3	| output 800									|
| Dropout            	| output 800   									|
| Fully connected		| output 43   									|

 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following parameters...
```
    EPOCHS = 30
    BATCH_SIZE = 128
    mu = 0
    sigma = 0.1
    learning_rate = 0.001
    keep_probability = 0.5 
```
#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 96.2%
* test set accuracy of 94.5%

If a well known architecture was chosen:
* What architecture was chosen? 
A modified LeNet architecture was chosen which fits the traffic sign classification target. The changes to LeNet included adding a dropout layer to avoid overfitting of the training data. Also, another convolution layer is added to enhance the classification done.
* Why did you believe it would be relevant to the traffic sign application?
Given LeNet function that was conducted in class, the result was around 89% of accuracy for the validation data. This was a good starting point to add the extra convolution layer, dropout layer (of 50% for training data), while adjusting the input training dataset by balancing the number of examples per class.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
When we calculated the Softmax for the images from the internet for traffic signs, this showed which predictions were the output of the neural network, and which predictions were formed for each sign with which percentage...
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web (fitted to 32x32 pixels image):

![alt text][image41] ![alt text][image42] ![alt text][image43] ![alt text][image44] 
![alt text][image45] ![alt text][image46] ![alt text][image47] ![alt text][image48]

The 50 Km/h road image might be difficult to classify because the numbers on the traffic sign are written in a font different from the one used in the training dataset. It looks like an old traffic sign, which is different from what the network trained to.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			                        |     Prediction	        					| 
|:-------------------------------------:|:---------------------------------------------:| 
| Stop Sign      		                | Stop sign   									| 
| Road Work    			                | Road work 									|
| Right of way at the next intersection | Right of way at the next intersection			|
| 50 km/h	      		                | 30  km/h     					 				|
| Go left	      		                | Go left      					 				|
| Slippery road			                | Slippery road      							|
| Turn right ahead		                | Turn right ahead     							|
| End of all speed and passing limits	| End of all speed and passing limits       	|



The model was able to correctly guess 7 of the 8 traffic signs, which gives an accuracy of 87.5%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the Softmax section of the Ipython notebook.

![alt text][image5]

As seen in the image, all images were correctly detected by the convolutional neural network except the 50 km/h sign. The font, and the angle are not exactly matching the standard German traffic signs. The image is taken from an old traffic sign, which is a bit different than the current training data in our dataset.


