
# A demo for Deep Transfer Hashing #
  
### Preprocessed data ###
- BBC
 - bbc_PCA_6_2of3.mat
 - bbc_PCA_6_1of3.mat
- Reuters
 - db_data_img.mat
 - db_data_tag.mat
### Prepare data for training and testing ###
* gen_data_reuters_bbc.py
### Model definition ###
* multilayer_perceptron.py
### Training script ###
* train_mlp.py
### Generate hash code for train/test data ###
* feed_model.py
### Evaluate performance ###
* /utils
* WTT.m
* hash_test.m

# Citation
*  Joey Tianyi Zhou, Heng Zhao, Xi Peng, Meng Fang, Zheng Qin and Rick Siow Mong Goh, "Transfer Hashing: From Shallow To Deep", Transaction on Neural Network and Learning Systems (TNNLS)
*  Joey Tianyi Zhou, Xinxing Xu, Sinno J. Pan, Ivor W. Tsang, Zheng Qin, and Rick Siow Mong Goh. "Transfer Hashing with Privileged Information,"  in Proceedings of the 25th International Joint Conference on Artificial Intelligence (IJCAI-16), NYC, USA, 2016.
