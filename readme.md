# Machine Learning for Gap Filling
The goal of this repository is showing how to use **data science** and, particularly, **Machine Learning** to predict future behavior.

For that I use a 1 year time serie of agromet data. This is the file PRE2013.xlsx.

I previously did this work in Matlab and published in 2014. At that time, Machine Learning algorithms were very "stripped", in the sense that you yourself had to do everything: create an ANN, test with different layers, choose the optimum neurons per layer, choose the non-linear function for each neuron, check the convergence of the gradient descent algorithm, ... Nowadays, using Python, mostly everything comes by default.

This brings the inconvenience of doing the stuff without knowing what it is going on under the hood, but, at least, it quite facilitates the use of Machine Learning. But in any way, that is another story...

Following the standard procedure, I splitted into 4 jupyter notebooks:
1. data_visualization nb, to plot some graphs just to get to know the data before any kind of processing.
2. data_preprocessing nb, to prepare the data for processing.
3. data_processing nb, in which I apply the regression model.
4. data_prediction nb, in which I use the model to predict future behavior and compare with the real data.

As a result, the model predicts acceptably well (R² = 0.6) the real data during the last month of 2013.
<img src="/assets/output.png"/>
