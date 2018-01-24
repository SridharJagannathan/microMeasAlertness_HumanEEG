function [outputLabel, accuracy, outputValue] = svmNFoldCrossValidation(label,data,run,cmd)

% We assume that label is ordered [1 1 1 1 2 2 2 2 3 3 3 3 3 4 4 4 4 4]';
% And the data and run is ordered accordingly
NClass = length(unique(label));
[N, D] = size(data);
outputLabel = zeros(size(label));
outputValue = zeros(size(label,1),NClass*(NClass-1)/2);

if size(run,1) == 1
    % Randomly pick a run number
    Ncv = run;
    run = [1:N]';
    run = mod(run,Ncv)+1;
    %run = run(randperm(N)); % shuffle the run number
elseif size(run,1) == N
    Ncv = length(unique(run));
end

for i = 1:Ncv
    trainIndex = run ~= i;
    trainData = data(trainIndex,:);
    trainLabel = label(trainIndex,:);
    
    testIndex = run == i;
    testData = data(testIndex,:);
    testLabel = label(testIndex,:);
% Train the model on the selected parameters
model = svmtrain(trainLabel, trainData, cmd);
% Classifying using the trained model
[predictedLabel, cv, decisValueWinner] = svmpredict(testLabel, testData, model); % run the SVM model on the test data
outputLabel(testIndex) = predictedLabel; 
outputValue(testIndex,:) = decisValueWinner;
end

accuracy = sum(outputLabel==label,1)/N;