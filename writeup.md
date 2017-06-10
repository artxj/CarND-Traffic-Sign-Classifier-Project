#**Traffic Sign Recognition**

##Writeup Template

---

**Build a Traffic Sign Recognition Project**

The steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Append the data set with 'artificial' data
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report
* Create a visualization of neural network layers


[//]: # (Image References)

[train_hist]: data:image/png;base64,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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup

Here is the link to the [project code](https://github.com/artxj/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) and to the [notebook html](https://github.com/artxj/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.html).

###Data Set Summary & Exploration

I used the numpy and matplotlib libraries to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because ...

I decided to generate additional data because ...

To add more data to the the data set, I used the following techniques because ...

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ...


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|



####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ?
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Stop Sign      		| Stop sign   									|
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .60         			| Stop sign   									|
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ...

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
