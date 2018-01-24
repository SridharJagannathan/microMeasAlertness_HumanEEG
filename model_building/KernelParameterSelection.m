function [bestc, bestg, bestcv] = KernelParameterSelection(trainLabel, trainData, Ncv, option)
% This function assist you to obtain the parameters C (c) and gamma (g)
% automatically.
%
% INPUT:
% trainLabel: An Nx1 vector denoting the label for each observation
% trainData: An N x D matrix denoting the feature/data matrix
% Ncv: A scalar representing Ncv-fold cross validation for parameter
% selection. Note that this function does not require the user to specify
% the run number for each iteration because it automatically assigns the run
% number in the code "get_cv_ac.m" (from the svmlib).
% option: options for parameters selecting
%
% OUTPUT:
% bestc: A scalar denoting the best value for C
% bestg: A scalar denoting the best value for g
% bestcv: the best accuracy calculated from the train data set
%
% Inspired from Kittipat "Bot" Kampa

% #######################
% Automatic Cross Validation 
% Parameter selection using n-fold cross validation
% #######################
[N, D] = size(trainData);

if nargin>3
    stepSize = option.stepSize;
    bestLog2c = log2(option.cmin);
    bestLog2g = log2(option.gammamin);
    bestLog2cmin = log2(option.cmin);
    bestLog2gmin = log2(option.gammamin);
    bestLog2cmax = log2(option.cmax);
    bestLog2gmax = log2(option.gammamax);
    epsilon = option.epsilon;
    svmCmd = option.svmCmd;
else
    msg = 'Incorrect number of arguments..';
    error(msg)
end

% initial some auxiliary variables
bestcv = 0;
deltacv = 10^6;
cnt = 1;
breakLoop = 0;

log2c_list = bestLog2cmin: stepSize: bestLog2cmax;
log2g_list = bestLog2gmin: stepSize: bestLog2gmax;

numLog2c = length(log2c_list);
numLog2g = length(log2g_list);

Nlimit = length(log2c_list)*length(log2g_list);
cv_collec =[];

while abs(deltacv) > epsilon && cnt < Nlimit
    bestcv_prev = bestcv;
    

    
    for i = 1:numLog2c
        log2c = log2c_list(i);
        for j = 1:numLog2g
            log2g = log2g_list(j);

            % With some precal kernel
            cmd = ['-c ', num2str(2^log2c), ' -g ', num2str(2^log2g),' ',svmCmd];
            [outputLabel, cv, outputValue] = svmNFoldCrossValidation(trainLabel, trainData, Ncv,cmd);
            cv_collec(i,j) = cv;
            
            if (cv >= bestcv),
                bestcv = cv; bestLog2c = log2c; bestLog2g = log2g;
                bestc = 2^bestLog2c; bestg = 2^bestLog2g;
            end
            disp(['So far, cnt=',num2str(cnt),' the best parameters, yielding Accuracy=',num2str(bestcv*100),'%, are: C=',num2str(bestc),', gamma=',num2str(bestg)]);
            % Break out of the loop when the cnt is up to the condition
            if cnt >= Nlimit, breakLoop = 1; break; end
            cnt = cnt + 1;
        end
        if breakLoop == 1, break; end
    end
    if breakLoop == 1, break; end
    deltacv = bestcv - bestcv_prev;
    
end
%disp(['The best parameters, yielding Accuracy=',num2str(bestcv*100),'%, are: C=',num2str(bestc),', gamma=',num2str(bestg)]);
cstr = ['2^(', num2str(log2(bestc)), ')'];
gstr = ['2^(', num2str(log2(bestg)), ')'];
disp(['The best parameters, yielding Accuracy=',num2str(bestcv*100),'%, are: C=',cstr,', gamma=',gstr]);

figure;
surf(log2g_list,log2c_list,cv_collec)
xlabel('gamma')
ylabel('C')
%cv_svm1 = cv_collec;
%save('cv_parameters_svm1','cv_svm1');



