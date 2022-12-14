function [trainedClassifier, validationAccuracy] = trainClassifier(trainingData)
% Extract predictors and response
% This code processes the data into the right shape for training the
% classifier.
inputTable = trainingData;
predictorNames = {'cough_detected', 'age', 'gender', 'respiratory_condition', 'fever_muscle_pain'};
predictors = inputTable(:, predictorNames);
response = inputTable.status;
isCategoricalPredictor = [false, false, true, true, true];

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
template = templateTree(...
    'MaxNumSplits', 20);
classificationEnsemble = fitensemble(...
    predictors, ...
    response, ...
    'AdaBoostM1', ...
    30, ...
    template, ...
    'Type', 'Classification', ...
    'LearnRate', 0.1, ...
    'ClassNames', {'COVID-19'; 'healthy'});

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
ensemblePredictFcn = @(x) predict(classificationEnsemble, x);
trainedClassifier.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = {'cough_detected', 'age', 'gender', 'respiratory_condition', 'fever_muscle_pain'};
trainedClassifier.ClassificationEnsemble = classificationEnsemble;
trainedClassifier.About = 'This struct is a trained classifier exported from Classification Learner R2016a.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedClassifier''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% classifier.
inputTable = trainingData;
predictorNames = {'cough_detected', 'age', 'gender', 'respiratory_condition', 'fever_muscle_pain'};
predictors = inputTable(:, predictorNames);
response = inputTable.status;
isCategoricalPredictor = [false, false, true, true, true];

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationEnsemble, 'KFold', 5);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');

% Compute validation predictions and scores
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);
