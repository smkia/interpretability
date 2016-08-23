%% Single Subject - Lasso
clear all;
addpath('/Functions');
lambda = [0.001,0.01,0.1,1,10,50,100,250,500,1000];
alpha = [1];
timeInterval = 76:325; % -200ms to 800ms
bootstrap_num = 50; %100;
    
for subj= 1 : 16
    
    % Preparing data
    filename = sprintf('"Path to data"/train_subject%02d.mat',subj);
    disp(strcat('Loading ',filename));
    data = load(filename);
    [trialNum,channelNum,timeNum] = size(data.X);
    data.y(data.y==0)=-1;
    data.X = data.X(:,:,timeInterval);
    d = reshape(data.X,trialNum,channelNum*length(timeInterval));
    d = mapstd(d')';
    target = single(data.y);
    clear data;
    [n] = size(d,1);
    
    Y_table = cell(length(lambda),length(alpha));
    acc = cell(length(lambda),length(alpha));
    for l = 1 : length(lambda)
        for a = 1 : length(alpha)
            Y_table{l,a} = nan(bootstrap_num,n);
            acc{l,a} = zeros(1,bootstrap_num);
        end
    end
    
    A = mean(d(target==1,:)) - mean(d(target==-1,:));
    mask = ones(channelNum,length(timeInterval));
    A = A.*reshape(mask',1,channelNum*length(timeInterval));
    performance = struct('UV',[],'BV',[],'BS',[],'VR',[],'EPE',[],'performance',[]);
    interpretable =  struct('representativeness',[],'reproducibility',[],'interpretability',[]);
    zeta = zeros(length(lambda),length(alpha));
    ACC = zeros(length(lambda),length(alpha));
    
    % Training
    for l = 1 : length(lambda)
        opts.lambda = lambda(l);
        for a = 1 : length(alpha)
            opts.alpha=alpha(a);
            [W,Y_table{l,a},acc] = OOB(d,target,bootstrap_num,opts,0);
            ACC(l,a) = mean(acc);
            [performance(l,a)] = EPE(Y_table{l,a},target);
            [interpretable(l,a)] = interpretability(W,A);
            zeta(l,a) = zeta_phi(performance(l,a).performance,interpretable(l,a).interpretability,1,1,0.6);
            disp(strcat('Subject:',num2str(subj),',Lambda:',num2str(lambda(l)), ...
                ',Alpha:',num2str(alpha(a)),',Performance:',num2str(performance(l,a).performance),...
                ',Interpretable:',num2str(interpretable(l,a).interpretability),',Zeta:',num2str(zeta(l,a))));
            save(strcat('subj',num2str(subj),'_ST_Lasso_Results.mat'),'Y_table','ACC','performance','zeta','interpretable','lambda','alpha');
        end
    end
end