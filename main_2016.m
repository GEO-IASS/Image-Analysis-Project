%% Created datastore object of dataset
myfolder = './Dataset';
imds = imageDatastore(myfolder, 'IncludeSubfolders',true,'LabelSource','foldernames');
tbl = countEachLabel(imds);
%% To Display Montage of few data
%sample = splitEachLabel(imds,8);
%montage(sample.Files(1:20));
%title(char(tbl.Label(1)));

%% Partition data set into 2 parts
%% Training set and test set

[training_set, test_set] = partition_data(imds, 8, 4);
return;
%% Create Visual Vocabulary 
tic
bag = bagOfFeatures(training_set,...
    'VocabularySize',250,'PointSelection','Detector');
scenedata = double(encode(bag, training_set));
toc
%% Visualize Feature Vectors 
%img = read(training_set(1), randi(training_set(1).Count));
%featureVector = encode(bag, img);

%subplot(4,2,1); imshow(img);
%subplot(4,2,2); 
%bar(featureVector);title('Visual Word Occurrences');xlabel('Visual Word Index');ylabel('Frequency');

%img = read(training_set(2), randi(training_set(2).Count));
%featureVector = encode(bag, img);
%subplot(4,2,3); imshow(img);
%subplot(4,2,4); 
%bar(featureVector);title('Visual Word Occurrences');xlabel('Visual Word Index');ylabel('Frequency');
%% 
SceneImageData = array2table(scenedata);
sceneType = categorical(repelem({training_set.Description}', [training_set.Count], 1));
SceneImageData.sceneType = sceneType;
%% Test out accuracy on test set!

testSceneData = double(encode(bag, test_set));
testSceneData = array2table(testSceneData,'VariableNames',trainedClassifier.RequiredVariables);
actualSceneType = categorical(repelem({test_set.Description}', [test_set.Count], 1));

predictedOutcome = trainedClassifier.predictFcn(testSceneData);

correctPredictions = (predictedOutcome == actualSceneType);
validationAccuracy = sum(correctPredictions)/length(predictedOutcome) %#ok
%% Visualize how the classifier works
ii = randi(size(test_set,2));
jj = randi(test_set(ii).Count);
img = read(test_set(ii),jj);

imshow(img)
% Add code here to invoke the trained classifier
imagefeatures = double(encode(bag, img));
% Find two closest matches for each feature
[bestGuess, score] = predict(trainedClassifier.ClassificationSVM,imagefeatures);
% Display the string label for img
if strcmp(char(bestGuess),test_set(ii).Description)
	titleColor = [0 0.8 0];
else
	titleColor = 'r';
end
title(sprintf('Best Guess: %s; Actual: %s',...
	char(bestGuess),test_set(ii).Description),...
	'color',titleColor)

return;