function [W,AP,A,Y_table,Y_table_AP,Y_table_A] = OOB2 (data,target,bootstrap_num,opts,parallel)
% This functions performs OOB procedure on a given dataset and model and
% returns weight of OOB models, their predictions, and accuracy. The
%
% Inputs:
%           data: input data organized as n*p samples where n is the number
%           samples and p is the number of features.
%           target: Label vector of data organized as n*1 vector. This
%           vector should contain 1 for positive and -1 for negative
%           classes.
%           bootstrap_num: number of bootstraps.
%           opts: parameters of the model.
%           parellel: 0 or 1. if one the code will run in parallel.
% Outputs:
%           W: is a 1*bootstrap_num cell that contains the weight vector of
%           the model in each bootstrap repitition.
%           AP: is a 1*bootstrap_num cell that contains the activation patterns of
%           the model in each bootstrap repitition.
%           A: is a 1*bootstrap_num cell that contains cERFs in each bootstrap repitition.
%           Y_table: is a bootstrap_num*n matrix that contains the
%           prediction of the model for all test samples in each run of
%           bootstrap.
%           Y_table_AP: is a bootstrap_num*n matrix that contains the
%           prediction computed by activation pattern for all test samples in each run of
%           bootstrap.
%           Y_table_A: is a bootstrap_num*n matrix that contains the
%           prediction of cERF for all test samples in each run of
%           bootstrap.

% Developed by Seyed Mostafa Kia (m.kia83@gmail.com)

[n] = size(data,1);
Y_table = nan(bootstrap_num,n);
Y_table_AP = nan(bootstrap_num,n);
Y_table_A = nan(bootstrap_num,n);
W = cell(1,bootstrap_num);
AP = cell(1,bootstrap_num);
A = cell(1,bootstrap_num);
y_pred =cell(1,bootstrap_num);
y_pred_a =cell(1,bootstrap_num);
y_pred_ap =cell(1,bootstrap_num);
randInd = zeros(bootstrap_num,n);
for f = 1 : bootstrap_num
    randInd(f,:) = randi(n,[1,n]);
end
if parallel
    parfor f = 1 : bootstrap_num
        X_tr = [];
        Y_tr = [];
        X_te = [];
        Y_te = [];
        X_tr = data(randInd(f,1:n),:);
        Y_tr = target(randInd(f,1:n));
        X_te = data(setdiff(1:n,randInd(f,:)),:);
        Y_te = target(setdiff(1:n,randInd(f,:)));
        [W{f}] = EN_LS(X_tr,Y_tr,opts);
        y_pred{f}= X_te*W{f};
        y_pred{f}= sign(y_pred{f});
        y_pred{f}(y_pred{f}==0) = 1;
        acc(f) = mean(Y_te==y_pred{f});
        disp(strcat('Bootstrap:',num2str(f)));
    end
else
    for f = 1 : bootstrap_num
        X_tr = [];
        Y_tr = [];
        X_te = [];
        Y_te = [];
        X_tr = data(randInd(f,1:n),:);
        Y_tr = target(randInd(f,1:n));
        X_te = data(setdiff(1:n,randInd(f,:)),:);
        Y_te = target(setdiff(1:n,randInd(f,:)));
        
        A{f} = mean(X_tr(Y_tr==1,:),1)-mean(X_tr(Y_tr==-1,:),1);
        A{f} = A{f}';
        y_pred_a{f}= X_te*A{f};
        y_pred_a{f}= sign(y_pred_a{f});
        y_pred_a{f}(y_pred_a{f}==0) = 1;
        
        [W{f}] = EN_LS(X_tr,Y_tr,opts);
        y_temp = X_tr*W{f};
        for k =  1 : size(X_tr,2)
            temp = cov([X_tr(:,k),y_temp]);
            AP{f}(k,1) = temp(1,2);
        end
        y_pred_ap{f}= sign(X_te*AP{f});
        y_pred_ap{f}(y_pred_ap{f}==0) = 1;
        
        y_pred{f}= X_te*W{f};
        y_pred{f}= sign(y_pred{f});
        y_pred{f}(y_pred{f}==0) = 1;
        disp(strcat('Bootstrap:',num2str(f)));
    end
end
for f = 1 : bootstrap_num
    Y_table(f,setdiff(1:n,randInd(f,:))) = y_pred{f};
    Y_table_A(f,setdiff(1:n,randInd(f,:))) = y_pred_a{f};
    Y_table_AP(f,setdiff(1:n,randInd(f,:))) = y_pred_ap{f};
end