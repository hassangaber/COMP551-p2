# COMP551 Mini-Project 2: SGD & Text Classification

## Setup

Run `pip install -r requirements.txt` to install all dependencies  
Run `pip freeze > requirements.txt` to update the requirements file

Include the datasets in the same directory as the repository in a folder named `data` with the `diabetes` and `fake_news` files inside

## Description
* *Part 1:* Observe gradient descent preformance as pramaters vary for the diabetes dataset
* *Part 2:* Preproccess fake-news data for binary classification model by mapping raw text details to text features and build a model to predict if text is human or computer-generated 

## Notes

* For the `diabetes` dataset, often times when feeding data to a model the input from the dataframe is interpreted as a 1D array inside of a 2D array. This causes obvious dimension incompatibility for dot products and such. To fix this issue: `x.to_numpy().ravel()` for the dataframe or dataframe fragment `x`.
* The decision boundary defined in the `optimize` function may be invalid, it defines it at X=0.5. It may need some tweaking.
* After implementing mini batch gradient descent, iterating though multiple epochs yielded no difference in preformance or accuracy, even the same weights.
* To fix the above problem, we have added a shuffle function to shuffle the data of the batches each epoch.

## Todo
- [x] Set up logistic regression and gradient descent functions from files
- [x] Find converging GD by optimizing max. iterations, learning rate, and training set accuracy on the test set
- [x] Implement mini-batch SGD
- [ ] Compare convergence speed and accuracy of mini-batch SGD with full SGD
- [x] Observe effect of varying the momentum coefficient in full SGD
- [ ] Accuracy plots for 1.3, 1.4
- [ ] Gradient oscillation graph with respect to solution
