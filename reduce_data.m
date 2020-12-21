%For PCA-reduced data with dimension k = 50, linear discriminant analysis training error rate is 7.8%.
%For PCA-reduced data with dimension k = 50, linear discriminant test error rate is 8.5%.
%For PCA-reduced data with dimension k = 50, perceptron training error rate is 7.8%.
%For PCA-reduced data with dimension k = 50, perceptron test error rate is 8.8%.
%For PCA-reduced data with dimension k = 100, linear discriminant analysis training error rate is 5.8%.
%For PCA-reduced data with dimension k = 100, linear discriminant test error rate is 10%.
%For PCA-reduced data with dimension k = 100, perceptron training error rate is 5.6%.
%For PCA-reduced data with dimension k = 100, perceptron test error rate is 10%.
%For PCA-reduced data with dimension k = 200, linear discriminant analysis training error rate is 4.4%.
%For PCA-reduced data with dimension k = 200, linear discriminant test error rate is 9.5%.
%For PCA-reduced data with dimension k = 200, perceptron training error rate is 4.5%.
%For PCA-reduced data with dimension k = 200, perceptron test error rate is 10%.
%For PCA-reduced data with dimension k = 400, linear discriminant analysis training error rate is 2.9%.
%For PCA-reduced data with dimension k = 400, linear discriminant test error rate is 10%.
%For PCA-reduced data with dimension k = 400, perceptron training error rate is 2.8%.
%For PCA-reduced data with dimension k = 400, perceptron test error rate is 10%.

%This function takes in a training data matrix Xtrain and uses
%it to compute the PCA basis and a sample mean vector. 
%It also takes in a test data matrix Xtest and a dimension k. 
%It first centers the data matrices Xtrain and Xtest by subtracting the
%Xtrain sample mean vector from each of their rows. It then uses the 
%top-k vectors in the PCA basis to project the centered Xtrain and Xtest
%data matrices into a k-dimensional space, and outputs
%the resulting data matrices as Xtrain_reduced and Xtest_reduced.
function [Xtrain_reduced Xtest_reduced] = reduce_data(Xtrain,Xtest,k)
V = pca(Xtrain);
Vk = V(1:size(V,1),1:k);
mutrain = sum(Xtrain,1)/size(Xtrain,1);

one_test = ones(size(Xtest,1),1);
one_train = ones(size(Xtrain,1),1);

Xtest_centered = Xtest - one_test * mutrain;
Xtrain_centered = Xtrain - one_train * mutrain;

Xtest_reduced = Xtest_centered * Vk;
Xtrain_reduced = Xtrain_centered * Vk;

end
