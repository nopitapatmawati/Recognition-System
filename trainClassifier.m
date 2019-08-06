function [trainedClassifier, validationAccuracy] = trainClassifier(trainingData)
% trainClassifier(trainingData)
%  returns a trained classifier and its accuracy.
%  This code recreates the classification model trained in
%  Classification Learner app.
%
%   Input:
%       trainingData: the training data of same data type as imported
%        in the app (table or matrix).
%
%   Output:
%       trainedClassifier: a struct containing the trained classifier.
%        The struct contains various fields with information about the
%        trained classifier.
%
%       trainedClassifier.predictFcn: a function to make predictions
%        on new data. It takes an input of the same form as this training
%        code (table or matrix) and returns predictions for the response.
%        If you supply a matrix, include only the predictors columns (or
%        rows).
%
%       validationAccuracy: a double containing the accuracy in
%        percent. In the app, the History list displays this
%        overall accuracy score for each model.
%
%  Use the code to train the model with new data.
%  To retrain your classifier, call the function from the command line
%  with your original data or new data as the input argument trainingData.
%
%  For example, to retrain a classifier trained with the original data set
%  T, enter:
%    [trainedClassifier, validationAccuracy] = trainClassifier(T)
%
%  To make predictions with the returned 'trainedClassifier' on new data T,
%  use
%    yfit = trainedClassifier.predictFcn(T)
%
%  To automate training the same classifier with new data, or to learn how
%  to programmatically train classifiers, examine the generated code.

% Auto-generated by MATLAB on 17-May-2018 06:39:47


% Extract predictors and response
% This code processes the data into the right shape for training the
% classifier.
inputTable = trainingData;
predictorNames = {'scenedata1', 'scenedata2', 'scenedata3', 'scenedata4', 'scenedata5', 'scenedata6', 'scenedata7', 'scenedata8', 'scenedata9', 'scenedata10', 'scenedata11', 'scenedata12', 'scenedata13', 'scenedata14', 'scenedata15', 'scenedata16', 'scenedata17', 'scenedata18', 'scenedata19', 'scenedata20', 'scenedata21', 'scenedata22', 'scenedata23', 'scenedata24', 'scenedata25', 'scenedata26', 'scenedata27', 'scenedata28', 'scenedata29', 'scenedata30', 'scenedata31', 'scenedata32', 'scenedata33', 'scenedata34', 'scenedata35', 'scenedata36', 'scenedata37', 'scenedata38', 'scenedata39', 'scenedata40', 'scenedata41', 'scenedata42', 'scenedata43', 'scenedata44', 'scenedata45', 'scenedata46', 'scenedata47', 'scenedata48', 'scenedata49', 'scenedata50', 'scenedata51', 'scenedata52', 'scenedata53', 'scenedata54', 'scenedata55', 'scenedata56', 'scenedata57', 'scenedata58', 'scenedata59', 'scenedata60', 'scenedata61', 'scenedata62', 'scenedata63', 'scenedata64', 'scenedata65', 'scenedata66', 'scenedata67', 'scenedata68', 'scenedata69', 'scenedata70', 'scenedata71', 'scenedata72', 'scenedata73', 'scenedata74', 'scenedata75', 'scenedata76', 'scenedata77', 'scenedata78', 'scenedata79', 'scenedata80', 'scenedata81', 'scenedata82', 'scenedata83', 'scenedata84', 'scenedata85', 'scenedata86', 'scenedata87', 'scenedata88', 'scenedata89', 'scenedata90', 'scenedata91', 'scenedata92', 'scenedata93', 'scenedata94', 'scenedata95', 'scenedata96', 'scenedata97', 'scenedata98', 'scenedata99', 'scenedata100', 'scenedata101', 'scenedata102', 'scenedata103', 'scenedata104', 'scenedata105', 'scenedata106', 'scenedata107', 'scenedata108', 'scenedata109', 'scenedata110', 'scenedata111', 'scenedata112', 'scenedata113', 'scenedata114', 'scenedata115', 'scenedata116', 'scenedata117', 'scenedata118', 'scenedata119', 'scenedata120', 'scenedata121', 'scenedata122', 'scenedata123', 'scenedata124', 'scenedata125', 'scenedata126', 'scenedata127', 'scenedata128', 'scenedata129', 'scenedata130', 'scenedata131', 'scenedata132', 'scenedata133', 'scenedata134', 'scenedata135', 'scenedata136', 'scenedata137', 'scenedata138', 'scenedata139', 'scenedata140', 'scenedata141', 'scenedata142', 'scenedata143', 'scenedata144', 'scenedata145', 'scenedata146', 'scenedata147', 'scenedata148', 'scenedata149', 'scenedata150', 'scenedata151', 'scenedata152', 'scenedata153', 'scenedata154', 'scenedata155', 'scenedata156', 'scenedata157', 'scenedata158', 'scenedata159', 'scenedata160', 'scenedata161', 'scenedata162', 'scenedata163', 'scenedata164', 'scenedata165', 'scenedata166', 'scenedata167', 'scenedata168', 'scenedata169', 'scenedata170', 'scenedata171', 'scenedata172', 'scenedata173', 'scenedata174', 'scenedata175', 'scenedata176', 'scenedata177', 'scenedata178', 'scenedata179', 'scenedata180', 'scenedata181', 'scenedata182', 'scenedata183', 'scenedata184', 'scenedata185', 'scenedata186', 'scenedata187', 'scenedata188', 'scenedata189', 'scenedata190', 'scenedata191', 'scenedata192', 'scenedata193', 'scenedata194', 'scenedata195', 'scenedata196', 'scenedata197', 'scenedata198', 'scenedata199', 'scenedata200', 'scenedata201', 'scenedata202', 'scenedata203', 'scenedata204', 'scenedata205', 'scenedata206', 'scenedata207', 'scenedata208', 'scenedata209', 'scenedata210', 'scenedata211', 'scenedata212', 'scenedata213', 'scenedata214', 'scenedata215', 'scenedata216', 'scenedata217', 'scenedata218', 'scenedata219', 'scenedata220', 'scenedata221', 'scenedata222', 'scenedata223', 'scenedata224', 'scenedata225', 'scenedata226', 'scenedata227', 'scenedata228', 'scenedata229', 'scenedata230', 'scenedata231', 'scenedata232', 'scenedata233', 'scenedata234', 'scenedata235', 'scenedata236', 'scenedata237', 'scenedata238', 'scenedata239', 'scenedata240', 'scenedata241', 'scenedata242', 'scenedata243', 'scenedata244', 'scenedata245', 'scenedata246', 'scenedata247', 'scenedata248', 'scenedata249', 'scenedata250'};
predictors = inputTable(:, predictorNames);
response = inputTable.sceneType;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
classificationKNN = fitcknn(...
    predictors, ...
    response, ...
    'Distance', 'Cosine', ...
    'Exponent', [], ...
    'NumNeighbors', 10, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', categorical({'Apple'; 'Non Apple'}));

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
knnPredictFcn = @(x) predict(classificationKNN, x);
trainedClassifier.predictFcn = @(x) knnPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = {'scenedata1', 'scenedata2', 'scenedata3', 'scenedata4', 'scenedata5', 'scenedata6', 'scenedata7', 'scenedata8', 'scenedata9', 'scenedata10', 'scenedata11', 'scenedata12', 'scenedata13', 'scenedata14', 'scenedata15', 'scenedata16', 'scenedata17', 'scenedata18', 'scenedata19', 'scenedata20', 'scenedata21', 'scenedata22', 'scenedata23', 'scenedata24', 'scenedata25', 'scenedata26', 'scenedata27', 'scenedata28', 'scenedata29', 'scenedata30', 'scenedata31', 'scenedata32', 'scenedata33', 'scenedata34', 'scenedata35', 'scenedata36', 'scenedata37', 'scenedata38', 'scenedata39', 'scenedata40', 'scenedata41', 'scenedata42', 'scenedata43', 'scenedata44', 'scenedata45', 'scenedata46', 'scenedata47', 'scenedata48', 'scenedata49', 'scenedata50', 'scenedata51', 'scenedata52', 'scenedata53', 'scenedata54', 'scenedata55', 'scenedata56', 'scenedata57', 'scenedata58', 'scenedata59', 'scenedata60', 'scenedata61', 'scenedata62', 'scenedata63', 'scenedata64', 'scenedata65', 'scenedata66', 'scenedata67', 'scenedata68', 'scenedata69', 'scenedata70', 'scenedata71', 'scenedata72', 'scenedata73', 'scenedata74', 'scenedata75', 'scenedata76', 'scenedata77', 'scenedata78', 'scenedata79', 'scenedata80', 'scenedata81', 'scenedata82', 'scenedata83', 'scenedata84', 'scenedata85', 'scenedata86', 'scenedata87', 'scenedata88', 'scenedata89', 'scenedata90', 'scenedata91', 'scenedata92', 'scenedata93', 'scenedata94', 'scenedata95', 'scenedata96', 'scenedata97', 'scenedata98', 'scenedata99', 'scenedata100', 'scenedata101', 'scenedata102', 'scenedata103', 'scenedata104', 'scenedata105', 'scenedata106', 'scenedata107', 'scenedata108', 'scenedata109', 'scenedata110', 'scenedata111', 'scenedata112', 'scenedata113', 'scenedata114', 'scenedata115', 'scenedata116', 'scenedata117', 'scenedata118', 'scenedata119', 'scenedata120', 'scenedata121', 'scenedata122', 'scenedata123', 'scenedata124', 'scenedata125', 'scenedata126', 'scenedata127', 'scenedata128', 'scenedata129', 'scenedata130', 'scenedata131', 'scenedata132', 'scenedata133', 'scenedata134', 'scenedata135', 'scenedata136', 'scenedata137', 'scenedata138', 'scenedata139', 'scenedata140', 'scenedata141', 'scenedata142', 'scenedata143', 'scenedata144', 'scenedata145', 'scenedata146', 'scenedata147', 'scenedata148', 'scenedata149', 'scenedata150', 'scenedata151', 'scenedata152', 'scenedata153', 'scenedata154', 'scenedata155', 'scenedata156', 'scenedata157', 'scenedata158', 'scenedata159', 'scenedata160', 'scenedata161', 'scenedata162', 'scenedata163', 'scenedata164', 'scenedata165', 'scenedata166', 'scenedata167', 'scenedata168', 'scenedata169', 'scenedata170', 'scenedata171', 'scenedata172', 'scenedata173', 'scenedata174', 'scenedata175', 'scenedata176', 'scenedata177', 'scenedata178', 'scenedata179', 'scenedata180', 'scenedata181', 'scenedata182', 'scenedata183', 'scenedata184', 'scenedata185', 'scenedata186', 'scenedata187', 'scenedata188', 'scenedata189', 'scenedata190', 'scenedata191', 'scenedata192', 'scenedata193', 'scenedata194', 'scenedata195', 'scenedata196', 'scenedata197', 'scenedata198', 'scenedata199', 'scenedata200', 'scenedata201', 'scenedata202', 'scenedata203', 'scenedata204', 'scenedata205', 'scenedata206', 'scenedata207', 'scenedata208', 'scenedata209', 'scenedata210', 'scenedata211', 'scenedata212', 'scenedata213', 'scenedata214', 'scenedata215', 'scenedata216', 'scenedata217', 'scenedata218', 'scenedata219', 'scenedata220', 'scenedata221', 'scenedata222', 'scenedata223', 'scenedata224', 'scenedata225', 'scenedata226', 'scenedata227', 'scenedata228', 'scenedata229', 'scenedata230', 'scenedata231', 'scenedata232', 'scenedata233', 'scenedata234', 'scenedata235', 'scenedata236', 'scenedata237', 'scenedata238', 'scenedata239', 'scenedata240', 'scenedata241', 'scenedata242', 'scenedata243', 'scenedata244', 'scenedata245', 'scenedata246', 'scenedata247', 'scenedata248', 'scenedata249', 'scenedata250'};
trainedClassifier.ClassificationKNN = classificationKNN;
trainedClassifier.About = 'This struct is a trained classifier exported from Classification Learner R2016b.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedClassifier''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% classifier.
inputTable = trainingData;
predictorNames = {'scenedata1', 'scenedata2', 'scenedata3', 'scenedata4', 'scenedata5', 'scenedata6', 'scenedata7', 'scenedata8', 'scenedata9', 'scenedata10', 'scenedata11', 'scenedata12', 'scenedata13', 'scenedata14', 'scenedata15', 'scenedata16', 'scenedata17', 'scenedata18', 'scenedata19', 'scenedata20', 'scenedata21', 'scenedata22', 'scenedata23', 'scenedata24', 'scenedata25', 'scenedata26', 'scenedata27', 'scenedata28', 'scenedata29', 'scenedata30', 'scenedata31', 'scenedata32', 'scenedata33', 'scenedata34', 'scenedata35', 'scenedata36', 'scenedata37', 'scenedata38', 'scenedata39', 'scenedata40', 'scenedata41', 'scenedata42', 'scenedata43', 'scenedata44', 'scenedata45', 'scenedata46', 'scenedata47', 'scenedata48', 'scenedata49', 'scenedata50', 'scenedata51', 'scenedata52', 'scenedata53', 'scenedata54', 'scenedata55', 'scenedata56', 'scenedata57', 'scenedata58', 'scenedata59', 'scenedata60', 'scenedata61', 'scenedata62', 'scenedata63', 'scenedata64', 'scenedata65', 'scenedata66', 'scenedata67', 'scenedata68', 'scenedata69', 'scenedata70', 'scenedata71', 'scenedata72', 'scenedata73', 'scenedata74', 'scenedata75', 'scenedata76', 'scenedata77', 'scenedata78', 'scenedata79', 'scenedata80', 'scenedata81', 'scenedata82', 'scenedata83', 'scenedata84', 'scenedata85', 'scenedata86', 'scenedata87', 'scenedata88', 'scenedata89', 'scenedata90', 'scenedata91', 'scenedata92', 'scenedata93', 'scenedata94', 'scenedata95', 'scenedata96', 'scenedata97', 'scenedata98', 'scenedata99', 'scenedata100', 'scenedata101', 'scenedata102', 'scenedata103', 'scenedata104', 'scenedata105', 'scenedata106', 'scenedata107', 'scenedata108', 'scenedata109', 'scenedata110', 'scenedata111', 'scenedata112', 'scenedata113', 'scenedata114', 'scenedata115', 'scenedata116', 'scenedata117', 'scenedata118', 'scenedata119', 'scenedata120', 'scenedata121', 'scenedata122', 'scenedata123', 'scenedata124', 'scenedata125', 'scenedata126', 'scenedata127', 'scenedata128', 'scenedata129', 'scenedata130', 'scenedata131', 'scenedata132', 'scenedata133', 'scenedata134', 'scenedata135', 'scenedata136', 'scenedata137', 'scenedata138', 'scenedata139', 'scenedata140', 'scenedata141', 'scenedata142', 'scenedata143', 'scenedata144', 'scenedata145', 'scenedata146', 'scenedata147', 'scenedata148', 'scenedata149', 'scenedata150', 'scenedata151', 'scenedata152', 'scenedata153', 'scenedata154', 'scenedata155', 'scenedata156', 'scenedata157', 'scenedata158', 'scenedata159', 'scenedata160', 'scenedata161', 'scenedata162', 'scenedata163', 'scenedata164', 'scenedata165', 'scenedata166', 'scenedata167', 'scenedata168', 'scenedata169', 'scenedata170', 'scenedata171', 'scenedata172', 'scenedata173', 'scenedata174', 'scenedata175', 'scenedata176', 'scenedata177', 'scenedata178', 'scenedata179', 'scenedata180', 'scenedata181', 'scenedata182', 'scenedata183', 'scenedata184', 'scenedata185', 'scenedata186', 'scenedata187', 'scenedata188', 'scenedata189', 'scenedata190', 'scenedata191', 'scenedata192', 'scenedata193', 'scenedata194', 'scenedata195', 'scenedata196', 'scenedata197', 'scenedata198', 'scenedata199', 'scenedata200', 'scenedata201', 'scenedata202', 'scenedata203', 'scenedata204', 'scenedata205', 'scenedata206', 'scenedata207', 'scenedata208', 'scenedata209', 'scenedata210', 'scenedata211', 'scenedata212', 'scenedata213', 'scenedata214', 'scenedata215', 'scenedata216', 'scenedata217', 'scenedata218', 'scenedata219', 'scenedata220', 'scenedata221', 'scenedata222', 'scenedata223', 'scenedata224', 'scenedata225', 'scenedata226', 'scenedata227', 'scenedata228', 'scenedata229', 'scenedata230', 'scenedata231', 'scenedata232', 'scenedata233', 'scenedata234', 'scenedata235', 'scenedata236', 'scenedata237', 'scenedata238', 'scenedata239', 'scenedata240', 'scenedata241', 'scenedata242', 'scenedata243', 'scenedata244', 'scenedata245', 'scenedata246', 'scenedata247', 'scenedata248', 'scenedata249', 'scenedata250'};
predictors = inputTable(:, predictorNames);
response = inputTable.sceneType;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationKNN, 'KFold', 7);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');

% Compute validation predictions and scores
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);