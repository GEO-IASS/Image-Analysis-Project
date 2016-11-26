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
% Macros for training set and test set
TRAINING_SET_SIZE = 8;
TEST_SET_SIZE = 4;
[training_set, test_set] = partition_data(imds, TRAINING_SET_SIZE, TEST_SET_SIZE);
%% Extract features from training set images 
tic
    bag = bagOfFeatures(training_set, 'VocabularySize', 250, 'PointSelection', 'Detector');
    fruitsdata = double(encode(bag, training_set));
toc
disp(randi(training_set(1).Count));

%% Visualize Feature Vectors 
%img = read(training_set(1), randi(training_set(1).Count));
%featureVector = encode(bag, img);
%subplot(4,2,1);
%imshow(img);
%subplot(4,2,2); 
%bar(featureVector);
%title('Visual Word Occurrences');
%xlabel('Visual Word Index');
%ylabel('Frequency');
%% 
fruitsImageData = array2table(fruitsdata);
replicated_elem = repelem({training_set.Description}', [training_set.Count], 1);
fruitType = categorical(replicated_elem);
fruitsImageData.fruitType = fruitType;
%% Before going forward , remember to choose best classifier from classification app then go ahead
%% Test out accuracy on test set!
[trainedClassifier, ~] = trainedClassifier(fruitsImageData);
testFruitsData = double(encode(bag, test_set));
testFruitsData = array2table(testFruitsData,'VariableNames',trainedClassifier.RequiredVariables);
actualFruitType = categorical(repelem({test_set.Description}', [test_set.Count], 1));

predictedOutcome = trainedClassifier.predictFcn(testFruitsData);

correctPredictions = (predictedOutcome == actualFruitType);
validationAccuracy = sum(correctPredictions)/length(predictedOutcome);
%% Visualize how the classifier works
warning('off', 'Images:initSize:adjustingMag');
ii = randi(size(test_set,2));
jj = randi(test_set(ii).Count);
img = read(test_set(ii),jj);

imshow(img)
% Add code here to invoke the trained classifier
imagefeatures = double(encode(bag, img));
% Find two closest matches for each feature
[bestGuess, score] = predict(trainedClassifier.ClassificationEnsemble,imagefeatures);
% Display the string label for img
if strcmp(char(bestGuess),test_set(ii).Description)
	titleColor = [0 0.8 0];
else
	titleColor = 'r';
end
title(sprintf('Best Guess: %s; Actual: %s',...
	char(bestGuess),test_set(ii).Description),...
	'color',titleColor)