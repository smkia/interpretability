%% Simulated MEG - Lasso
clear all;
addpath('Functions');
addpath(genpath('Path/to/Fieldtrip'))
lambda = [0.001,0.01,0.1,1,10,50,100,250,500,1000];
alpha = [1];
bootstrap_num = 50;
iterNum = 25;    

cfgSimulation.sourcePos = [-4.7 -3.7 5.3];
cfgSimulation.sourceMom = [1,1,0];
cfgSimulation.samplingRate = 300;
cfgSimulation.trialNumPerClass = 250;
cfgSimulation.trialLength = 100;
cfgSimulation.channelType = 'MEGMAG';
cfgSimulation.freq = [3,5];
cfgSimulation.jitter = 3;
cfgSimulation.bpfilter = [0.3,45];

trialNum = 2 * cfgSimulation.trialNumPerClass;

zeta = zeros(iterNum,length(lambda),length(alpha));
ACC = zeros(iterNum,length(lambda),length(alpha));

for iter = 1 : iterNum
    % Preparing data
    [raw,raw_GT] = MEG_simulation(cfgSimulation);
    [channelNum,timeNum] = size(raw{1}.trial{1});
    X = zeros(trialNum,channelNum*timeNum);
    Y = zeros(trialNum,1);
    for i = 1 : cfgSimulation.trialNumPerClass
        X(i,:) = reshape(raw{1}.trial{i},[1,channelNum*timeNum]);
        Y(i) = 1;
    end
    for i = 1 : cfgSimulation.trialNumPerClass
        X(cfgSimulation.trialNumPerClass+i,:) = reshape(raw{2}.trial{i},[1,channelNum*timeNum]);
        Y(cfgSimulation.trialNumPerClass+i) = -1;
    end
    %clear raw
    X = single(mapstd(X')');
    Y = single(Y);
        
    Y_table = cell(length(lambda),length(alpha));
    acc = cell(length(lambda),length(alpha));
    for l = 1 : length(lambda)
        for a = 1 : length(alpha)
            Y_table{l,a} = nan(bootstrap_num,trialNum);
            acc{l,a} = zeros(1,bootstrap_num);
        end
    end
    
    A{iter} = mean(X(Y==1,:)) - mean(X(Y==-1,:));
    A{iter} = A{iter}/norm(A{iter});
    
    GT = reshape(raw_GT.trial{1},[channelNum*timeNum,1]);
    GT = GT/norm(GT);
    
    % Training
    for l = 1 : length(lambda)
        opts.lambda = lambda(l);
        for a = 1 : length(alpha)
            opts.alpha=alpha(a);
            [W,AP,A,Y_table{iter,l,a},Y_table_AP{iter,l,a},Y_table_A{iter,l,a}] = OOB2(X,Y,bootstrap_num,opts,0);
            [performance(iter,l,a)] = EPE(Y_table{iter,l,a},Y);
            [performance_AP(iter,l,a)] = EPE(Y_table_AP{iter,l,a},Y);
            [interpretable_tilde(iter,l,a)] = interpretability(W,A{iter});
            [interpretable(iter,l,a)] = interpretability(W,GT);
            [interpretable_AP(iter,l,a)] = interpretability(AP,GT);
            zeta(iter,l,a) = zeta_phi(performance(iter,l,a).performance,interpretable_tilde(iter,l,a).interpretability,1,1,0.6);
            disp(strcat('Iter:',num2str(iter),',Lambda:',num2str(lambda(l)), ...
                ',Alpha:',num2str(alpha(a)),',Performance:',num2str(performance(iter,l,a).performance),...
                ',Interpretable:',num2str(interpretable(iter,l,a).interpretability),',Zeta:',num2str(zeta(iter,l,a))));
        end
    end
    save(strcat('SimulatedMEG_Lasso_Results.mat'),'Y_table','ACC','performance','zeta','interpretable','lambda','alpha'...
                ,'performance_AP','interpretable_tilde','interpretable_AP','GT','A');
end