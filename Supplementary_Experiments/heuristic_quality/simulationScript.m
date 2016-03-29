%%  Experiment 1
clear all;
iterations = 15;
sampleNum = [20,200,500,1000,2000,5000,10000,15000];
uncertainty_Y = [0,0.01,0.05,0.1,0.2,0.3]; % Epsilon
uncertainty_X = [1,1,1,1,1,1];  % Sigma

effectSize = [100,100];
freq = 5;
time=0.01:0.01:1/freq;
sinWave = sin(2*pi*time*freq).*hann(length(time))';
% creating the effect
effect = zeros(effectSize);
for i = 41 : 50
    effect(i,41:60) = effect(i,41:60) + sinWave;
end
for i = 51:60
    effect(i,41:60) = effect(i,41:60) + fliplr(sinWave);
end

addpath('/Functions')
bootstrap_num = 50;
Theta_star = reshape(effect,[1,effectSize(1)*effectSize(2)]);
opts.lambda = 0;
opts.alpha= 1;

for u = 1 : length(uncertainty_Y)
    for s = 1 : length(sampleNum)
        for iter = 1 : iterations
            data = single(zeros(sampleNum(s),effectSize(1),effectSize(2)));
            % Class A
            for i = 1 : sampleNum(s)/2
                noise = randn(effectSize(1),effectSize(2));
                data(i,:,:) = effect + uncertainty_X(u).*noise;
                S(i) = snr(noise,effect);
            end
            % Class B
            for i = sampleNum(s)/2+1 : sampleNum(s)
                data(i,:,:) = uncertainty_X(u).*randn(effectSize(1),effectSize(2));
            end
            SNR(u,s,iter) = mean(S);
            features = single(zeros(sampleNum(s),effectSize(1)*effectSize(2)));
            for i = 1 : sampleNum(s)
                features(i,:) = reshape(squeeze(data(i,:,:)),[1,effectSize(1)*effectSize(2)]);
            end
            clear data;
            target = zeros(sampleNum(s),1);
            target(1:sampleNum(s)/2,1)=1;
            target(sampleNum(s)/2+1:end,1)=-1;
            rand_idx = randi(sampleNum(s),[round(uncertainty_Y(u)*sampleNum(s)),1]);
            for i = 1 : length(rand_idx)
                if target(rand_idx(i)) == 1
                     target(rand_idx(i)) = -1;
                else
                     target(rand_idx(i)) = 1;
                end
            end
            Theta_IBDS = mean(features(target==1,:)) - mean(features(target==-1,:));
            d = mapstd(features')';
            % Classification
            l = 1;
            a = 1;
            Delta_beta(u,s,iter) = 1 - abs(pdist([Theta_IBDS;Theta_star],'cosine'));
            Y_table = cell(1,1);
            [W,Y_table{1,1},acc] = OOB (d,target,bootstrap_num,opts,1);
            ACC(u,s,iter) = mean(acc);
            [performance{u,s,iter}(l,a)] = EPE(Y_table{l,a},target{1});
            [interpretable_star{u,s,iter}(l,a)] = interpretability(W,Theta_star);
            plausible_star{u,s,iter}(l,a) = (performance{u,s,iter}(l,a).performance+interpretable_star{u,s,iter}(l,a).interpretability)/2;
            [interpretable_IBDS{u,s,iter}(l,a)] = interpretability(W,Theta_IBDS);
            plausible_IBDS{u,s,iter}(l,a) = (performance{u,s,iter}(l,a).performance+interpretable_IBDS{u,s,iter}(l,a).interpretability)/2;
            disp(strcat('Uncertainty:',num2str(uncertainty_Y(u)),',Sample Size:',num2str(sampleNum(s)),',Iter:',num2str(iter),',Performance:',num2str(performance{u,s,iter}(l,a).performance),...
                ',Interpretable:',num2str(interpretable_star{u,s,iter}(l,a).interpretability),',Plausible:',num2str(plausible_star{u,s,iter}(l,a))));
        end
    end
    save(strcat('Simulation_Results_Yuncertain_ST.mat'),'performance','interpretable_star','plausible_star','interpretable_IBDS','plausible_IBDS','SNR','sampleNum','Delta_beta','uncertainty_Y','uncertainty_X','ACC');
end

%%  Experiment 2
clear all;
iterations = 15;
sampleNum = [20,200,500,1000,2000,5000,10000,15000];
uncertainty_Y = [0,0,0,0,0,0]; % Epsilon
uncertainty_X = [0,0.25,0.75,1,1.5,2];  % Sigma

effectSize = [100,100];
freq = 5;
time=0.01:0.01:1/freq;
sinWave = sin(2*pi*time*freq).*hann(length(time))';
% creating the effect
effect = zeros(effectSize);
for i = 41 : 50
    effect(i,41:60) = effect(i,41:60) + sinWave;
end
for i = 51:60
    effect(i,41:60) = effect(i,41:60) + fliplr(sinWave);
end

addpath('/Functions')
bootstrap_num = 50;
Theta_star = reshape(effect,[1,effectSize(1)*effectSize(2)]);
opts.lambda = 0;
opts.alpha= 1;

for u = 1 : length(uncertainty_Y)
    for s = 1 : length(sampleNum)
        for iter = 1 : iterations
            data = single(zeros(sampleNum(s),effectSize(1),effectSize(2)));
            % Class A
            for i = 1 : sampleNum(s)/2
                noise = randn(effectSize(1),effectSize(2));
                data(i,:,:) = effect + uncertainty_X(u).*noise;
                S(i) = snr(noise,effect);
            end
            % Class B
            for i = sampleNum(s)/2+1 : sampleNum(s)
                data(i,:,:) = uncertainty_X(u).*randn(effectSize(1),effectSize(2));
            end
            SNR(u,s,iter) = mean(S);
            features = single(zeros(sampleNum(s),effectSize(1)*effectSize(2)));
            for i = 1 : sampleNum(s)
                features(i,:) = reshape(squeeze(data(i,:,:)),[1,effectSize(1)*effectSize(2)]);
            end
            clear data;
            target = zeros(sampleNum(s),1);
            target(1:sampleNum(s)/2,1)=1;
            target(sampleNum(s)/2+1:end,1)=-1;
            rand_idx = randi(sampleNum(s),[round(uncertainty_Y(u)*sampleNum(s)),1]);
            for i = 1 : length(rand_idx)
                if target(rand_idx(i)) == 1
                     target(rand_idx(i)) = -1;
                else
                     target(rand_idx(i)) = 1;
                end
            end
            Theta_IBDS = mean(features(target==1,:)) - mean(features(target==-1,:));
            d = mapstd(features')';
            % Classification
            l = 1;
            a = 1;
            Delta_beta(u,s,iter) = 1 - abs(pdist([Theta_IBDS;Theta_star],'cosine'));
            Y_table = cell(1,1);
            [W,Y_table{1,1},acc] = OOB (d,target,bootstrap_num,opts,1);
            ACC(u,s,iter) = mean(acc);
            [performance{u,s,iter}(l,a)] = EPE(Y_table{l,a},target{1});
            [interpretable_star{u,s,iter}(l,a)] = interpretability(W,Theta_star);
            plausible_star{u,s,iter}(l,a) = (performance{u,s,iter}(l,a).performance+interpretable_star{u,s,iter}(l,a).interpretability)/2;
            [interpretable_IBDS{u,s,iter}(l,a)] = interpretability(W,Theta_IBDS);
            plausible_IBDS{u,s,iter}(l,a) = (performance{u,s,iter}(l,a).performance+interpretable_IBDS{u,s,iter}(l,a).interpretability)/2;
            disp(strcat('Uncertainty:',num2str(uncertainty_X(u)),',Sample Size:',num2str(sampleNum(s)),',Iter:',num2str(iter),',Performance:',num2str(performance{u,s,iter}(l,a).performance),...
                ',Interpretable:',num2str(interpretable_star{u,s,iter}(l,a).interpretability),',Plausible:',num2str(plausible_star{u,s,iter}(l,a))));
        end
    end
    save(strcat('Simulation_Results_Xuncertain_ST.mat'),'performance','interpretable_star','plausible_star','interpretable_IBDS','plausible_IBDS','SNR','sampleNum','Delta_beta','uncertainty_Y','uncertainty_X','ACC');
end

