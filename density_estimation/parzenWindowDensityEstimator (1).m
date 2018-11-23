function [estimatedDensity] = parzenWindowDensityEstimator(trainingData,testingData,windowWidth)
% trainingdata should be a matrix of n rows and (d+1) columns
% The first d columns of trainingdata correspond to the d features
% The last column ( (d+1)-th ) corresponds to the class labels.
% The class labels should be from 1 to number of classes
%
% For testingData, there is NOT any class label
% So, if you have m testing data points, you should pass a m by d matrix
% where the i-th row corresponds to the i-th testing point, and the j-th
% column corresponds to the j-th feature.
%
% windowWidth is the width of the parzen window
%
% Note that it is using a different form of the Parzen window here.
% For the purpose of the assignment, you can simply substitute 
% the specified width parameter 
%
% For output, estimateDensity(j, c) corresponds to the estimated Parzen window
% class conditional density, p(y_j|\omega_c), for the j-th testing point 
% y_j from class c.
%


classLabels = trainingData(:,end);
numFeatures = size(trainingData,2)-1;
numTrainingData = size(trainingData, 1);
numClasses = length(unique(classLabels));
trainingData = trainingData(:,1:numFeatures);
testingSize = size(testingData,1);
classSamplesArray = cell(1,numClasses);

for i=1:numClasses
    indexOfClassI = find(classLabels == i);
    samplesOfClassI = trainingData(indexOfClassI,:);
    classSamplesArray{1,i} = samplesOfClassI;
    sigmaOfClassI = estimateVariance(samplesOfClassI);
    sigmaArray{1,i} = sigmaOfClassI;
end

estimatedDensity = zeros(testingSize,numClasses);

for i=1:testingSize
    x = testingData(i,:);
    for j=1:numClasses
        samplesOfClassJ = classSamplesArray{1,j} ;
        sigmaOfClassJ = sigmaArray{1,j};
        trainingSizeOfClassJ = size(samplesOfClassJ,1);
        testSampleMatrix = ones(trainingSizeOfClassJ,1)*x;
        
        new_diff = (testSampleMatrix-samplesOfClassJ);
        
        for k=1:numFeatures
            new_diff(abs(new_diff(:,k))>windowWidth, k) = 10000000000; %big number;
        end
        
        estimatedDensity(i,j) = mean((1/(windowWidth^numFeatures))*mvnpdf((new_diff/windowWidth),zeros(1,numFeatures),sigmaOfClassJ));
    end
end

%
% So, it is using            1/(windowwidth^d) * Normal(0, \sigma_j / window_width)
% as the window function for class j, instead of the spherical parzen window function
%

function sigma = estimateVariance(samples)

numFeatures = size(samples,2);
sigma = zeros(numFeatures,numFeatures);
for i=1:numFeatures
    sigma(i,i)=var(samples(:,i));
end
