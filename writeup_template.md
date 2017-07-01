#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is [32 32]
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the distribution of the types of road signs being shown. One important nugget is that there are some classes (like 21, or double curve) that are missing.
![alt text]['./writeup-files/histogram.png']


###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I tried to normalize the data, but doing so seemed to make the images extremely noisy. Grayscaling the images took forever and yielded a degradation in my model accuracy. I ended up ditching any efforts to preprocess the images.  I have left some comments to demonstrate what I tried, but the majority of the work I tried has been deleted. I probably spent way too much time trying to preprocess, and I believe that if my attempts were more successful, my model would have been much better.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|											    |
| Avg pooling	      	| 2x2 stride, outputs 14x14x6   				|
| Convolution 3x3       | 1x1 stride, same padding, outputs 12x12x16    |
| RELU                  |                                               |
| Avg pooling	      	| 2x2 stride, outputs 6x6x16    				|
| Flatten               | 6x6x16 to 576     	    					|
| Flat Neural Network	| 576->400->200->100->43   						|
| Dropout    			| During training, null out half activations	|
| Softmax				| Turn our logits into probablities				|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a model that closely resembles the LeNet architecture. I used a smaller filter in the second layer. I was hoping that this would allow the neural net to 'see' the smaller details that are inside of the road sign. I also decided to use average pooling, as this resulted in a better model for me. With a 2x2 stride for both of the pooling operations, I managed to make limit the dimensions of the two convolutional networks. I tried tweaking the learning rate, but I did not have much success so I left it at 0.001.

The batch size of 32 seemed to be an ideal mix of performance and accuracy. I decided to increase the number of epochs because this gave me more confidence that the validation accuracy was plateauing to a healthy number.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.4% Can be seen in code block [373] 
* validation set accuracy of 93.2% [304]
* test set accuracy of 90.9% [371]

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I chose LeNet because it is a solid image classifer that already has a healthy mix of convolutional layers, and the ability to include a dropout layer at the end.
* What were some problems with the initial architecture?
I believe that the size of the filters were a little large, and that the flattened neural net was not extensive enough.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I made the filter in the second layer smaller, to try to 'see' more specific feature from the images. I changed the pooling to average pooling. Because my filter was smaller, my flattened output was a little smaller which let me add more layers to the flattened neural network. I believe that the smaller filter size allowed me to mitigate any overfitting that would happen from having such a high number of flat layers. I also used a dropout layer to mitigate any overfitting. 
* Which parameters were tuned? How were they adjusted and why?
I tuned the amount of epochs because I was achieving 93% on the 10th epoch and I wanted to make sure this wasn't a fluke. Extending it allowed to me validate that the model plateaued at greater than 93%
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Having the two convolutional layers helped because it allows the classifier to be succesful regardless of the rotation, position, or noisyness of the road sign. The average pooling allows us to contain the amount of dimensions while also detecting feature patterns. The dropout layer at the very end allowed me to mitigate overfitting.

 
###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:


I downloaded these images from a tourist website that explained road signs in Germany. They are not real images, but rather stock images. I had to resize them which resulted in loss of detail and possibly some artificial noisyness.
![alt text][./web-examples/doublecurve.jpg] ![alt text][./web-examples/pedestrians.jpg] ![alt text][./web-examples/roadwork.jpg] 
![alt text][./web-examples/traffficsignals.jpg] ![alt text][./web-examples/warning.jpg]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Double Curve     		| Children Crossing 							| 
| Pedestrians  			| Pedestrians 				    				|
| Road Work				| Road Work										|
| Traffic Signals	    | Traffic Signals				 				|
| General Caution		| General Caution     							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is not bad, especially since the reason that the double curve failed was because there is no training data on double curve. I don't know if this is because the data is non-existent or if I made a mistake while loading it. This only differs from the test test accuracy by 10%. The other issue is that I decided to use stock images with no noise in them. Any noise that the images have are a result of resizing them to 32x32 images. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in cell[350] of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were


####Double Curve
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 3.9%         			| Children Crossing							    | 
| 3.6%    				| Beware of ice/snow 							|
| 3.6%					| Right-of-way at the next intersection		    |

####Pedestrians 
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 26.7%         		| Pedestrians	    				            | 
| 19.8%    				| General caution 		     					|
| 7.83%					| Right-of-way at the next intersection		    |

####Road Work 
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 99.4%         		| Road Work			                 		    | 
| -0.5%    				| General Caution	        					|
| -7.65%				| Dangerous curve to the right	        	    |

####Traffic Signals
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 56.6%         		| Traffic Signals		            		    | 
| 38.7%    				| General Caution 	       						|
| -28.42%				| Road Work	                               	    |

####General Caution 
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 96.8%         			| General Caution						    | 
| -10.4%    				| Traffic signals							|
| -53.2%					| Speed limit (60km/h)		                |


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


