clc;
clear;

imds = imageDatastore('D:\Semester 6\Sisrek\TUBES\DATASET','IncludeSubFolders',true,'LabelSource','foldernames');
tr = countEachLabel(imds);
categories = tr.Label;
[train_data, test_data]=prepareInputFiles(imds);

save('Dataset.mat');

bag = bagOfFeatures(train_data, 'vocabularySize',250,'PointSelection','Detector');
scenedata=double(encode(bag, train_data));

SceneImageData = array2table(scenedata);
sceneType = categorical(repelem({train_data.Description}', [train_data.Count], 1));
SceneImageData.sceneType=sceneType;

% classificationLearner
[trainedClassifier, TrainingAccuracy] = trainClassifier(SceneImageData);
save('trainedClassifier.mat')

testSceneData = double(encode(bag, test_data));
testSceneData = array2table(testSceneData,'VariableNames',trainedClassifier.RequiredVariables);
actualSceneType = categorical(repelem({test_data.Description}', [test_data.Count], 1));

predictedOutcome = trainedClassifier.predictFcn(testSceneData);

correctPredictions = (predictedOutcome == actualSceneType);
TestingAccuracy = sum(correctPredictions)/length(predictedOutcome);

ii = randi(size(test_data,2));
jj = randi(test_data(ii).Count);
img = read(test_data(ii),jj);

imshow(img)
% Add code here to invoke the trained classifier
imagefeatures = double(encode(bag, img));
imgSceneData = array2table(imagefeatures,'VariableNames',trainedClassifier.RequiredVariables);
% Find two closest matches for each feature
[bestGuess, score] = trainedClassifier.predictFcn(imgSceneData);
% [bestGuess, score] = predict(fitcknn(img, imagefeatures),imagefeatures);
% Display the string label for img
if strcmp(char(bestGuess),test_data(ii).Description)
	titleColor = [0 0.8 0];
else
	titleColor = 'r';
end
title(sprintf('Best Guess: %s; Actual: %s',...
	char(bestGuess),test_data(ii).Description),...
	'color',titleColor)
