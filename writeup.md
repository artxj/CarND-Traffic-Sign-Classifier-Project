## Traffic Sign Recognition

---

**Traffic Sign Recognition Project**

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

[all]: ./all.jpg
[train_hist]: ./images/train_hist.png
[train_hist_after]: ./images/train_hist_after.png
[30kmh]: ./images/30kmh.png

[sign1]: ./web-data/01.jpg
[sign2]: ./web-data/02.jpg
[sign3]: ./web-data/03.jpg
[sign4]: ./web-data/04.jpg
[sign5]: ./web-data/05.jpg
[sign6]: ./web-data/06.jpg
[sign7]: ./web-data/07.jpg
[sign8]: ./web-data/08.jpg
[sign9]: ./web-data/09.jpg

[pred01]: ./images/pred01.png
[pred02]: ./images/pred02.png
[pred03]: ./images/pred03.png
[pred04]: ./images/pred04.png
[pred05]: ./images/pred05.png
[pred06]: ./images/pred06.png
[pred07]: ./images/pred07.png
[pred08]: ./images/pred08.png
[pred09]: ./images/pred09.png

[conv1_1]: ./images/conv_sample1.png
[conv2_1]: ./images/conv2_sample1.png
[conv3_1]: ./images/conv3_sample1.png

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

Here is the link to the [project code](./Traffic_Sign_Classifier.ipynb) and to the [notebook html](./report.html).

### 1. Data Set Summary & Exploration

I used the numpy and matplotlib libraries to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

### 2. Exploratory visualization of the dataset.

All possible signs are the following:
![][all]

Here is an exploratory visualization of the data set. This is a bar chart showing the train data histogram.

![][train_hist]

It is easy to see that the data is unbalanced and some classes contain much more data than another ones.

### 3. Design and Test a Model Architecture

#### Preparing the data

As a first step, I've recognized signs that stay the same if flipped vertically (for example, 30kmh, 80kmh, priority road etc.) or horizontally (yield, bumpy road etc.). Some of signs can be flipped horizontally, then vertically and the result belongs to the same class - like 'End of all limits' sign. Also, there are signs that can be flipped horizontally and the result will be of an opposite class - for example, 'Turn left' sign after flipping becomes 'Turn right'.

As example, one of 30kmh signs flipped vertically looks like
![][30kmh]

I applied the listed changes to the training set and I got 58529 images instead of 34799 original ones. However, the classes still stay unbalanced:

![][train_hist_after]

Also, simple function was used to normalize all pixel color values to [-1; 1]:
```
new_value = (old_value - 127.5)/127.5
```

I also thought of converting images to grayscale but decided that color might be useful in some cases for specific signs (like priority road which is the only yellow sign in the set).

#### Model architecture

The final model consists of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x8 	|
| RELU					|												|
| Dropout     	| Probability of 0.8 	|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					|												|
| Dropout     	| Probability of 0.8 	|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 14x14x32 	|
| RELU					|												|
| Dropout     	| Probability of 0.8 	|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x64 	|
| RELU					|												|
| Dropout     	| Probability of 0.8 	|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 5x5x128 	|
| RELU					|												|
| Dropout     	| Probability of 0.8 	|
| Fully connected		| 1024 neurons        									|
| RELU					|												|
| Dropout     	| Probability of 0.5 	|
| Softmax				| 43 outputs        									|


#### Training the model

Adam optimizer was used as the most common on practice. Batch size was 128 and number of epochs was 150. Starting learning rate was set as 0.001 and it was decreased to 0.0001 after 50 epochs. L2 regularization was used for calculating the loss with beta parameter of 0.0001.

#### Model results

My final model results were:
* training set accuracy of 0.99983
* validation set accuracy of 0.989
* test set accuracy of 0.975

The best result after several runs (without saving checkpoints between) was 0.992 on the validation set that give 0.981 on the test one.

As a first step, I implemented simple LeNet architecture and by adjusting parameters I got 0.96 on validation set. After that I've played with adding and removing layers. It turns out that several fully connected layers doesn't play big role here so only one can be left. Also, very quick increase of filters depth for convolution layers also doesn't give good results so I've decided of increasing the depth as 3 > 8 > 16 > 32 > 64 > 128 with max pooling after every 2 convolution layers. Without dropouts and without L2 regularization it gave the validation accuracy of about 0.970 approximately. But dropout with probability less than 0.7 for convolution layers results in low validation set accuracy, so I separated probabilities for full connected layer and convolution layers dropouts.

It's interesting that the model without adding 'artificial' data gives the best validation accuracy of 0.992 and it gives 0.971 on the test set.


### Test a Model on New Images

Here are ten German traffic signs found on the web:

![][sign1] ![][sign2] ![][sign3] ![][sign4] ![][sign5] ![][sign6] ![][sign7] ![][sign8] ![][sign9]

Notice that 4th sign is very far away (yet easily recognizable because of its shape and color), 7th sign has faded digits and 9th sign is also far and hard to recognize even by human eye.

The results of the prediction are:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Speed limit (120km/h)      		| Speed limit (120km/h)   									|
| Keep right     			| Keep right 										|
| Wild animals crossing					| Wild animals crossing											|
| Priority road	      		| Ahead only					 				|
| No vehicles			| No vehicles      							|
| End of speed limit (80km/h)      		| End of speed limit (80km/h) sign   									|
| Speed limit (120km/h)     			| Speed limit (120km/h) 										|
| Dangerous curve to the right					| Stop											|
| Speed limit (80km/h)      		| Speed limit (80km/h)					 				|

The model correctly predicted 7 of 9 signs which gives and accuracy of 77.8%. The model was unable to recognize two cases with far signs, however it correctly classified a difficult case with 120km/h sign (faded digits).

I've also run a predictions on a model without artificial data (see previous chapter) with same validation accuracy. It's interesting that it was unable to predict End of speed limit (80km/h) so adding new data actually helped here.

The top five soft max probabilities of predictions for the listed images are the following ones:

![][sign1]
![][pred01]

![][sign2]
![][pred02]

![][sign3]
![][pred03]

![][sign4]
![][pred04]

![][sign5]
![][pred05]

![][sign6]
![][pred06]

![][sign7]
![][pred07]

![][sign8]
![][pred08]

![][sign9]
![][pred09]

It's interesting to notice that both priority road and dangerous curve weren't even in top five for that signs. Also, the model is not so sure about second speed limit 120km/h sign due to its difficult recognizing even by human eye.


### Visualizing the Neural network

Here are some samples from convolution layers 1, 2 and 3 after applying ReLU activation function to the image of 'Keep right':

![][sign2]

![][conv1_1]
![][conv2_1]
![][conv3_1]

Interesting to notice how things get simpler when going to lower levels. Also, some masks are completely black which means they decline all input pixels for the given image.
