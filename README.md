
## Abstract
This report solves a two-class pattern classification problem with Radial Basis Function (RBF) neural network and Support Vector Machines (SVM) neural network, respectively. Classifiers are trained using the given training data, and the performance of two classifiers with different parameters is discussed in this report.

## 1.	Data
The given training data consists of 330 samples, and each piece is represented by a one-dimensional array of length 33. The validation data is selected from training data randomly, and the selected data will not be applied to the training process. There are two kinds of labels in the training label: -1 and 1. The testing data contains 21 samples.

## 2.	RBF Neural Network  
In the training process of the Radial Basis Function neural network, centers need to be determined first. There are three typical strategies for neuron center determination:  
__(1)__	Random selection from training samples  
__(2)__	Prototypes of training samples as neuron centers  
__(3)__	Center selection as a model selection problem  
In this report, I apply the first two methods to determine the center. I assume the number of neurons in this section is fixed. I will discuss how the performance of the classifier changes with the number of neurons in 4.2.

### 2.1 Random Center
___Parameters:___  
a.	Basis function: Gaussian basis function  
b.	Center_x = Center_y = 8 (Assume that the centers are distributed in matrix form)  
c.	Sigma:   = 0.875  
___Results:___    
By verifying the model thirty times through the validation set, I calculate the average MSE and accuracy shown in Table 1.  

### 2.2	Prototype Center
There are two typical clustering algorithms: Self-Organizing Map(SOM) and K-Means clustering. Both of them could be applied to prototype center selection.

### 2.2.1	Self-Organizing Map (SOM)
___Parameters:___ <br />
a.	Weight initialization: Randomly take 33 floating-point numbers in the range of -1 to 1 as an initial center sample.<br />
b.	Basis function: Gaussian basis function<br />
c.	Center_x = Center_y = 8 <br />
d.	Learning rate η: At iteration n, we have η(n) = η0 * exp(-n/τ2). η0 = 0.1, τ2 = 1000.<br />
e.	Sigma: At iteration n, we have σ(n)=σ0 * exp(-n/τ1). 𝜎0 = dmax/2 = 4.95, τ1 = 1000/ln 𝜎0 = 625<br />
f.	Iteration time: 1000<br />
___Results:___    
By verifying the model thirty times through the validation set, I calculate the average MSE and accuracy shown in Table 1.

### 2.2.2	K-Means
___Parameters:___ <br />
a.	Basis function: Gaussian basis function<br />
b.	Center number: 64 <br />
c.	Sigma: 1<br />
___Results:___    
By verifying the model thirty times through the validation set, I calculate the average MSE and accuracy shown in Table 1.

### 3.	SVM Neural Network
___Parameters:___ <br />
a.	Kernel Functions: Gaussian kernel function<br />
b.	Decision function shape: one v one <br />
c.	C: 1.0<br />
d.	Gamma: 1.0<br />
___Results:___    
By verifying the model thirty times through the validation set, I calculate the average MSE and accuracy shown in Table 1.

| __Classifier__ | __Average MSE__ | __Average Accuracy__ | 
| ------ | ------ | ------ | 
| __RBF-Random Center__ | 0.6321 | 0.8787 | 
| __RBF-SOM Center__ | 0.5431 | 0.8453 | 
| __RBF-K-Means Center__ | 0.4532 | 0.8820 | 
| __SVM__ | 0.3510 | 0.9123 | 

Table 1: Performance of RBF neural network with different strategies for neuron center determination and SVM neural network.

## 4.	Data Visualization
### 4.1 Performance and stability 
Since the validation set is randomly selected and RBF neural network has different centers every time, the performance of these neural networks has a degree of randomness. To observe the performance of these neural networks more intuitively and their stability to random data, I recorded the MSE and accuracy of each time of the thirty calculations and plotted the image with the MSE and accuracy recorded as scatter points. (Show in Figure1)

### 4.2 Number of RBF Hidden Layer Neurons
In order to compare the performance of different center selection strategies more precisely, the number of neurons in the second part is set to 64 for all designs. To find the suitable number of hidden layer neurons in the RBF classifier, I take the RBF model generated by K-MEANS as an example. I change the number of neurons from 26 to 100 and calculate each digit ten times to reduce the error caused by randomness to get the average of the MSE and accuracy. Scattered points and fitted straight lines are shown in Figure 2.<br />
![](https://github.com/Accumul/EE7207Assignment/blob/master/Images/Figure1.png)  
Figure 1: MSE and accuracy performance plot
![](https://github.com/Accumul/EE7207Assignment/blob/master/Images/Figure2.png)  
Figure 2: Changes in performance with changes in neuron numbers 

### 4.3 Overfitted Problem
I only focus on the classifier change when the number of neurons increases in a small range in the last part. In this part, I compared the performance of the validation set and training set with the increase of hidden layer neuron numbers in the RBF-KMeans classifier. I use accuracy as the criterion and observe its change as the number of centers increases from 1 to 200. The line chart is shown in Figure 3.<br />
![](https://github.com/Accumul/EE7207Assignment/blob/master/Images/Figure%203.png)  
Figure 3: Accuracy for train set and validation set-Neuron number Relationship

## 5.	Conclusion
For the Radial Basis Function neural network, both strategies for neuron center determination and the number of hidden layer neurons will affect the performance of the classifier. <br />
1.	The random center strategy shows instability. Some MSE values will exceed 1, while some are about 0.3. The reason for the fluctuation is some random centers are more suitable while some are not.<br />
2.	The performance of the prototype center is much better than the random one, and the classifier trained using this strategy is relatively stable. In this strategy, the K-Means clustering is more accurate than the SOM clustering as a whole. <br />
3.	In the RBF-KMeans model, the MSE gradually decreases, and the accuracy gradually increases with the number of neurons increases from 26 to 100. It indicates that increasing the number of hidden layer neurons within a reasonable range helps to train a classifier with better performance.<br />
As the neuron increases in a larger range, for example, from 1 to 200, the accuracy of the validation set reached its peak at 95% when the number of neurons was around 100. Then, the accuracy of the validation set begins to decline, while that of the training set is still increasing, and the accuracy reached almost 100% at the end. I think this is a manifestation of data overfitting.<br />

Support Vector Machines neural network showed lower stability but higher overall average performance than RBF neural network (Prototype center strategy). 
