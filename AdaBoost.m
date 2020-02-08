function [train_err, test_err] = AdaBoost(X_tr, y_tr, X_te, y_te, numTrees)
% AdaBoost: Implement AdaBoost with decision stumps as the weak learners. 
%           (hint: read the "Name-Value Pair Arguments" part of the "fitctree" documentation)
%   Inputs:
%           X_tr: Training data
%           y_tr: Training labels
%           X_te: Testing data
%           y_te: Testing labels
%           numTrees: The number of trees to use
%  Outputs: 
%           train_err: Classification error of the learned ensemble on the training data
%           test_err: Classification error of the learned ensemble on test data
% 
% You may use "fitctree" but not any inbuilt boosting function
numRows = size(X_tr,1);
testRows = size(X_te,1);
weights = zeros(numRows,1);
weights(:,1)=(1/numRows);

ythreeIndex = find(y_tr==3);
yfiveIndex = find(y_tr==5);

ytestIndex = find(y_te==3);
ytestFive = find(y_te==5);

y_tr(ythreeIndex) = -1;
y_te(ytestIndex) = -1;
y_tr(yfiveIndex) = 1;
y_te(ytestFive) = 1;

alpha = 0;

agg_tErr = zeros(numRows,1);
agg_test = zeros(testRows,1);

trainErrorArr = zeros(numTrees,1);
testErrorArr = zeros(numTrees,1);

error = 0;
test_error = 0;
epsilon = 0;
for i=1:numTrees
    
    trainTree = fitctree(X_tr,y_tr,'MaxNumSplits', 1, 'Weights', weights);
    
    
    trainPredict = predict(trainTree, X_tr);
    
    
    testPredict = predict(trainTree, X_te);
    
    
    epsilon = 0;
    %Calculating Epsilon
    for j=1:numRows
        if(trainPredict(j)~=y_tr(j))
            epsilon = epsilon + weights(j);
        end
    end
    
    %Calculating the Alpha value based on Epsilon
    alpha = (1/2)*log((1-epsilon)/epsilon);
    
    %Calculating the aggregated Training Error based on alpha & Predictions
    agg_tErr = agg_tErr + alpha*trainPredict;
    
    %Calculating the Aggregated Test Error based on alpha and Predictions
    agg_test = agg_test + alpha*testPredict;
    
    %Calculating the zeta value
    zeta = 2*sqrt(epsilon*(1-epsilon));
    
    %Updating Weights if predictions don't equal y
    for x = 1:numRows
      if trainPredict(x)~=y_tr(x)
          weights(x) = (weights(x)*(exp(alpha))/zeta);
      end
    end
    
    %Updating Weights if predictions = y
    for o = 1:numRows
        if trainPredict(o)==y_tr(o)
          weights(o) = (weights(o)*(exp(-alpha))/zeta);
        end
    end
    
    %calculating Error based on if agg does not equal y
    error = 0;
    for j=1:numRows
        if sign(agg_tErr(j)) ~= y_tr(j)
            error = error +1;         
        end
    end
    
    %Calculating test error based on sign of aggregated test error
    test_error = 0;
    for k=1:testRows
        if sign(agg_test(k)) ~= y_te(k)
            test_error = test_error +1;         
        end
    end
    
    %Finding the average test error
    test_error = test_error/testRows;
    
    %Average training error
    error = error/numRows;
    
    %Storing both errors
    trainErrorArr(i) = error;
    
    testErrorArr(i) = test_error;
    
end
%Array from 1 to 200 to plot the oob error
treeArray = linspace(1,numTrees,200);


train_err = trainErrorArr(200);


test_err = testErrorArr(200);


plot(treeArray, trainErrorArr);

hold on;

plot(treeArray,testErrorArr);
legend('Training Error','Test Error')
hold off;
end