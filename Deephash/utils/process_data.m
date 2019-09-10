function XX_normalized = process_data(data)

averageNumberNeighbors = 50;
[num, d] = size(data);
% define ground-truth neighbors (this is only used for the evaluation):
num = randperm(num); 
DtrueTraining = distMat(data(num(1:100), :), data); % sample 100 points to find a threshold
Dball = sort(DtrueTraining, 2);    %DtrueTraining sort by row
clear DtrueTraining;
Dball = mean(Dball(:, averageNumberNeighbors));

% scale data so that the target distance is 1
data = data / Dball;

% generate training ans test split and the data matrix
XX = data;

% center the data, VERY IMPORTANT
sampleMean = mean(XX,1);
XX = (double(XX)-repmat(sampleMean,size(XX,1),1));

% normalize the data
%XX_normalized = normalize1(XX);
XX_normalized = XX;


