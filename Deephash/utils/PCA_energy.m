function PCA_X = PCA_energy (ratio,pca_X_all,eigVal)

% pca_X_all----- d*n
sum_PCA = sum(eigVal);
current_PCA = 0;
d_PCA = 1;
while (current_PCA < ratio*sum_PCA)
    current_PCA = current_PCA + eigVal (d_PCA);
    d_PCA = d_PCA +1;
end
PCA_X = pca_X_all(:,1:d_PCA);