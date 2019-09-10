function [ProjectMatrix, EigenValues] = perform_pca(features)
% [ProjectMatrix EigenValues] = perform_pca()
%
% Compute the PCA projection matrix, support sparse matrix for fast
%
% Input: 
%   features : d-by-n matrix
% Output:
%   ProjectMatrix, EigenValues
%
% by Joey Tianyi Zhou on 25 Sep, 2012
% -------------------
% revised by Joey Tianyi Zhou on 9 Oct, 2013
% - do not sub mean at the beginning, for saving memory
% - Note, only tested kernel case.

tic;
[dim, N]    = size(features);
mean_feat   = mean(features, 2);
% features    = features - repmat(mean_feat, 1, N);
% % after sub mean, it is not sparse, so we full it to parallelize
% features    = full(features);
tt = toc;
fprintf('\tPCA:feature preprocssing time = %f\n', tt);

if dim <= N
    fprintf('\tDim < N, do cov decomposition\n');
    tic;
    cov = features*features' - N*mean_feat*mean_feat';
    cov = full(cov);
    tt = toc;
    fprintf('\tPCA:cov computing time = %f\n', tt);
    tic;
    [eigVec eigVal] = eig(cov);
    tt = toc;
    fprintf('\tPCA:eig computing time = %f\n', tt);
else    
    fprintf('\tDim > N, do kernel decomposition\n');
    tic;    
    tmp = (mean_feat'*features);
    cov = features'*features - tmp'*ones(1, N) - ones(N, 1)*tmp + ones(N)*(mean_feat'*mean_feat);
    cov = full(cov);
    tt = toc;
    fprintf('\tPCA:cov computing time = %f\n', tt);
    tic
    [eigVec eigVal] = eig(cov);
    tmp     = sum(eigVec);
    eigVec  = features*eigVec;
    eigVec  = eigVec - mean_feat*tmp;
    eigVec  = eigVec ./ repmat(sqrt(sum(eigVec.^2)), [dim, 1]);    %normalize
    tt = toc;
    fprintf('\tPCA:eig computing time = %f\n', tt);
end
[EigenValues ind]   = sort(diag(eigVal), 'descend');
ProjectMatrix       = eigVec(:, ind);


% 
% 
% if nFeatureDim < nTotalSample
%     nTDim = nFeatureDim;
%     DataCov = DataWithoutMean * DataWithoutMean'; % d-by-d
%     [eigVector, eigValue] = eig(DataCov);
%     for i = 1 : nTDim
%         eigVector(:, i) = eigVector(:, i) / norm(eigVector(:, i));
%     end
% else
%     nTDim = nTotalSample;
%     DataCov = full(DataWithoutMean' * DataWithoutMean);
%     [eigVector_temp, eigValue] = eig(DataCov);
%     eigVector = DataWithoutMean * eigVector_temp; 
% %     eigVector = eigVector_temp;
%     %normalize each eigen vector
%     for i = 1 : nTDim
%         eigVector(:, i) = eigVector(:, i) / norm(eigVector(:, i));
%     end;
% end;
% 
% 
% X = diag(eigValue);
% [Y, nIndex] = sort(-X);
% EIGs = -Y;
% 
% eigenVector_s = eigVector(:, nIndex);
% 
% PCAProjectionMatrix = eigenVector_s;
% 
% end