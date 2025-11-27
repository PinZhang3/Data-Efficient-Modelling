clear;  
clc;   
close all;

% l1qc_logbarrier is available from http://www.acm.caltech.edu/l1magic/

% This code serves as an exemplar of model development in the paper "Data-Efficient Characterisation of Soil Variability through Active Learning and Knowledge-Informed Multi-Fidelity Modelling", authored by Geng-Fu HE and Pin ZHANG 

% The code should NOT be distributed without prior permission from the authors before publication.

count = 1;  

N_orig = 512;
len = 20;   

mse_bay = zeros(1,len); 
mse_bp = zeros(1,len);  
mse_fast = zeros(1,len); 

rce_bay = zeros(1,len); 
rce_bp = zeros(1,len);
rce_fast = zeros(1,len); 

Cc_bay = zeros(1,len);  
Cc_bp  = zeros(1,len);
Cc_fast = zeros(1,len);

load Files;

% Random Phi
y_target_all=y_target_all;
CPTdata_all=CPTdata_all;
N_orig=size(CPTdata_all, 2);

%% Initial
rng(29); 
y_location_index(:,2)=0;
random_vectors= randperm(size(y_target_all,1)) 
Num_initial=3  
random_initial_data=random_vectors(1:Num_initial)

%%Sampling
N_RS_datasets =  7;  
Plan_N_AL_datasets =  7; 
max_std_index_array = zeros(1, N_RS_datasets+1);

y_location_index(random_initial_data,2)=1;  

Recording_RE=[]
Recording_MCI=[]   
Recording_RI=[]
Recording_N_datsets=[]

Cell_x_BP_matrix = cell(50, 1);

for Random_sampling  = 1 : N_RS_datasets+1  

%data
observed_index = find(y_location_index(:,2)>0);  
y_target=y_target_all(observed_index,:);

%basis vectors normalization
CPTdata_all = CPTdata_all ./ vecnorm(CPTdata_all, 2); 

CPTdata=CPTdata_all(observed_index,:);

y_location=y_location_index(:,1);
y_train_location=y_location_index(observed_index,1);

% Adding correlation using ro_mat 
phi_orig = 1*CPTdata;  

sig_n = 0.001;   
alpha_0 = 1/sig_n;  
alpha = 0.01*ones(1,N_orig); 
pos_indx = 1 : N_orig; 

% Generating x and obtaining y

Number_train=size(y_target,1); 
Ensemble_num=200 
x_BP_matrix = zeros(size(CPTdata_all, 2), Ensemble_num); 

% Bootstraping 
for trials  = 1 : 1 : Ensemble_num   
idx = randi(Number_train, Number_train, 1); 
subset_idx = unique(idx);
y = y_target(subset_idx,:);  
phi = phi_orig(subset_idx,:);
K=length(subset_idx)
%% Basis pursuit  
x0 = phi'*inv(phi*phi')*y; 
epsilon =  sig_n*sqrt(K)*sqrt(1 + 2*sqrt(2)/sqrt(K)); 
x_BP = l1qc_logbarrier(x0, phi, [], y, epsilon, 1e-6); 
x_BP_matrix(:, trials) = x_BP;  
Mean_weights=mean(x_BP_matrix, 2); 
end

temporary_prediction=CPTdata_all*x_BP_matrix;
Cell_x_BP_matrix{Random_sampling+Num_initial-1} = x_BP_matrix;

% Sampling
temporary_CI(observed_index)=0  ;
candidate_elements=setdiff(1:length(y_location_index(:,2)), observed_index);
rs_std_index=candidate_elements(randi(length(candidate_elements)));
y_location_index(rs_std_index,2)=1;  
max_std_index_array(Random_sampling) = rs_std_index;

end