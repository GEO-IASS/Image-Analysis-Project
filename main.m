%% Created datastore object of dataset
myfolder = './Dataset';
imds = imageDatastore(myfolder, 'IncludeSubfolders',true,'LabelSource','foldernames');
tbl = countEachLabel(imds);
minSetCount = min(tbl{:,2});
%% To Display Montage of few data
%sample = splitEachLabel(imds,8);
%montage(sample.Files(1:20));
%title(char(tbl.Label(1)));

%% Partition data set into 2 parts
%% Training set and test set
% Macros for training set and test set
TRAINING_SET_SIZE = 2;
TEST_SET_SIZE = 2;
[training_set, test_set] = partition_data(imds, TRAINING_SET_SIZE, TEST_SET_SIZE);
% Extract features from training set images 
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
% 
fruitsImageData = array2table(fruitsdata);
replicated_elem = repelem({training_set.Description}', [training_set.Count], 1);
fruitType = categorical(replicated_elem);
fruitsImageData.fruitType = fruitType;

%% Test out accuracy on test set!

[trainedClassifier, ~] = random(fruitsImageData);
testFruitsData = double(encode(bag, test_set));
testFruitsData = array2table(testFruitsData,'VariableNames',trainedClassifier.RequiredVariables);
actualFruitType = categorical(repelem({test_set.Description}', [test_set.Count], 1));
predictedOutcome = trainedClassifier.predictFcn(testFruitsData);
correctPredictions = (predictedOutcome == actualFruitType);
validationAccuracy = sum(correctPredictions)/length(predictedOutcome);
disp(validationAccuracy);
%% Get File From Our User and show classifier best Guess
warning('off', 'Images:initSize:adjustingMag');
user_input_file = imgetfile;
% ii = randi(size(test_set,2));
% jj = randi(test_set(ii).Count);4
img = imread(user_input_file);
imshow(img)
imagefeatures = double(encode(bag, img));
% Find two closest matches for each feature
[bestGuess, score] = predict(trainedClassifier.ClassificationSVM,imagefeatures);
title(sprintf('Best Guess: %s;',char(bestGuess)));

%%
a = [2 4 3 6];
c = csvread('test.csv');
s = struct('v', c);
% r = readtable('test.csv');
% s = table2struct(r);
% % % load('go');
[bestGuess, score] = predict(trainedClassifier.ClassificationTree,s.v);
title(sprintf('Best Guess: %s;',char(bestGuess)));
