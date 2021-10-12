# COMP551 Mini-Project 2: SGD & Text Classification

## Description
* *Part 1:* Observe gradient descent preformance as pramaters vary for the diabetes dataset
* *Part 2:* Preproccess fake-news data for binary classification model by mapping raw text details to text features and build a model to predict if text is human or computer-generated 

## Notes:
* Include the datasets in the same directory as the repository in a folder named `data` with the `diabetes` and `fake_news` files inside
* For the `diabetes` dataset, often times when feeding data to a model the input from the dataframe is interpreted as a 1D array inside of a 2D array. This causes obvious dimension incompatibility for dot products and such. To fix this issue: `x.to_numpy().ravel()` for the dataframe or dataframe fragment `x`.
* The decision boundary defined in the `optimize` function may be invalid, it defines it at X=0.5. It may need some tweaking.

## TODO:
- [x] Set up logistic regression and gradient descent functions from files
- [ ] Find converging GD solution by optimizing learning rates and maximum iterations of the GD function
- [ ] Find relationship between gradient normal and validation accuracy as a function of learning rate and maximum iterations
