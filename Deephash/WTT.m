function wtt = WTT(db_train_data, db_test_data)

% 1,000 data points are random selected from the whole data set 
% as the queries, and the remaining is used to form the gallery database
% and treated as the targets for search. A nominal threshold of the average 
% distance to the 50th nearest neighbor is used to determine whether a 
% database point returned for a given query is considered a true positive.

addpath('./utils/');

% parameters
averageNumberNeighbors = 50;    % ground truth is 50 nearest neighbor

num_train = size(db_train_data, 1);
train_data = db_train_data;
test_data = db_test_data;

% define ground-truth neighbors (this is only used for the evaluation):
R = randperm(num_train); 
DtrueTraining = distMat(train_data(R(1:100), :), train_data); % sample 100 points to find a threshold
Dball = sort(DtrueTraining, 2);    %DtrueTraining sort by row
clear DtrueTraining;
Dball = mean(Dball(:, averageNumberNeighbors));

% scale data so that the target distance is 1
train_data = train_data / Dball;
test_data = test_data / Dball;
Dball = 1;

% threshold to define ground truth
DtrueTestTraining = distMat(test_data, train_data);
WtrueTestTraining = DtrueTestTraining < Dball;
clear DtrueTestTraining;


wtt = WtrueTestTraining;

%save(wtt, 'WTT');


%fprintf('calculating WTT for %s database has finished\n\n', db_name);