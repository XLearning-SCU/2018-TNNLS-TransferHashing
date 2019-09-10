close all; clear all; clc;

%% load data

data_folder = 'data_tmp';

original_feature_train = sprintf('.\\%s\\data_train_chosen_img.mat',data_folder);
original_feature_test = sprintf('.\\%s\\data_test_chosen_img.mat',data_folder);
hash_code_file_train = sprintf('.\\%s\\data_train_chosen_bins.mat',data_folder);
hash_code_file_test = sprintf('.\\%s\\data_test_chosen_bins.mat',data_folder);

load (original_feature_train); 
load (original_feature_test); 
load (hash_code_file_train); 
load (hash_code_file_test); 

%% test parameter
query_ID = [];
param.choice = 'evaluation';

loopnbits = [64];%[8 16 32 64];%[8 16 32 64 128];
runtimes = 3;    % modify it more times such as 8 to make the rusult more precise
choose_bits = 1;    % i: choose the bits to show for evaluation
choose_times = 1;    % k is the times of run times to show for evaluation

param.pos = [1 10:10:40 50:50:500];    % The number of retrieved samples: Recall-The number of retrieved samples curve

% FOR bbc 1828 samples total
% FOR reuters over 16k samples total
num_train_all = size(data_train_chosen_img,1);
% num_test_all = size(data_test_chosen_img,1);
num_test_all = 1000;

n_all = num_train_all + num_test_all;

param.num_test = num_test_all;
pos = param.pos;

%% evaluation
fprintf('cal start\n');
drawnow('update');
tic;
for k = 1:runtimes
  
    fprintf('runtimes: %d\n',k);
    drawnow('update');
    test_idx_all_perm = randperm(num_test_all);
    test_idx = test_idx_all_perm(1:param.num_test);
    WtrueTestTraining = WTT(data_train_chosen_img, data_test_chosen_img(test_idx,:));

    for i =1:length(loopnbits)      
      Dhamm = hammingDist(uint16(data_test_chosen_bins(test_idx,:)), uint16(data_train_chosen_bins));      
      [recall_t, precision_t, ~] = recall_precision(WtrueTestTraining, Dhamm);
      [rec_t, pre_t]= recall_precision5(WtrueTestTraining, Dhamm, pos); % recall VS. the number of retrieved sample
      [mAP_t] = area_RP(recall_t, precision_t);
      mAP_all(k) = mAP_t;
      recall{k}{i} = recall_t;
      precision{k}{i} = precision_t;
      mAP{k}{i} = mAP_t;
      rec{k}{i} = rec_t;
      pre{k}{i} = pre_t;
    end
    
end
t=toc
mean(mAP_all)
mAP_all
nhmethods = 1;
choose_bits = 1;
[~,choose_times] = max(mAP_all);

%% plot attribution
line_width = 2;
marker_size = 8;
xy_font_size = 14;
legend_font_size = 12;
linewidth = 1.6;
title_font_size = xy_font_size;

%% show precision vs. the number of retrieved sample.
figure('Color', [1 1 1]); hold on;
posEnd = 8;
for j = 1: nhmethods
    pos = param.pos;
    prec = pre{choose_times}{j};
    p = plot(pos(1,1:end), prec(1,1:end));
    color = gen_color(j);
    marker = gen_marker(j);
    set(p,'Color', color);
    set(p,'Marker', marker);
    set(p,'LineWidth', line_width);
    set(p,'MarkerSize', marker_size);
end

str_nbits =  num2str(loopnbits(choose_bits));
set(gca, 'linewidth', linewidth);
h1 = xlabel('The number of retrieved samples');
h2 = ylabel(['Precision @ ', str_nbits, ' bits']);
title('bbc', 'FontSize', title_font_size);
set(h1, 'FontSize', xy_font_size);
set(h2, 'FontSize', xy_font_size);
grid minor
%axis square;
hashmethods = {'MLP'}; 
hleg = legend(hashmethods);
set(hleg, 'FontSize', legend_font_size);
box on; grid on; hold off;
