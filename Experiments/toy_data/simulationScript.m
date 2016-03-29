%%  Simulating the data
clear all;
iterations = 10;
sampleNum = 1000;
p = 2;
S1 = [1.02,-.3;-.3,0.15];
S2 = [1.02,-.3;-.3,0.15];
m1 = [1.5;0];
m2 = [-1.5;0];

addpath(genpath('/Functions/'));
lambda = [0,0.001,0.01,0.1,1,10,50,100,250,500,1000,5000,10000,15000,25000,50000];
alpha = [1];
bootstrap_num = 50; %100;
bootstrap_frac = 1; 
shuffle_Y = 0;

for iter = 1 : iterations
    x1 = chol(S1)'*randn(p,round(sampleNum))+repmat(m1,1,round(sampleNum));
    x2 = chol(S2)'*randn(p,round(sampleNum))+repmat(m2,1,round(sampleNum));
    X = [x1';x2'];
    Y = [ones(round(sampleNum),1);-ones(round(sampleNum),1)];
    if shuffle_Y
        Y = Y(randperm(length(Y)));
    end
   
    % Classification
    [n] = size(X,1);
    Y_table = cell(length(lambda),length(alpha));
    acc = cell(length(lambda),length(alpha));
    for l = 1 : length(lambda)
        for a = 1 : length(alpha)
            Y_table{l,a} = nan(bootstrap_num,n);
            acc{l,a} = zeros(1,bootstrap_num);
        end
    end
    Theta_IBDS = mean(X(Y==1,:)) - mean(X(Y==-1,:));
    Theta_Star = m1' - m2';
    % Training
    for l = 1 : length(lambda)
        opts.lambda = lambda(l);
        for a = 1 : length(alpha)
            opts.alpha=alpha(a);
            [W,Y_table{l,a},acc] = OOB (X,Y,bootstrap_num,opts);
            ACC(l,a) = mean(acc);
            [performance{iter}(l,a)] = EPE(Y_table{l,a},Y);
            [interpretable_star{iter}(l,a)] = interpretability(W,Theta_Star);
            plausible_star{iter}(l,a) = zeta_phi(performance{iter}(l,a).performance,interpretable_star{iter}(l,a).interpretability,1,1,0.6);
            [interpretable_IBDS{iter}(l,a)] = interpretability(W,Theta_IBDS);
            plausible_IBDS{iter}(l,a) = zeta_phi(performance{iter}(l,a).performance,interpretable_IBDS{iter}(l,a).interpretability,1,1,0.6);
            disp(strcat('Iter:',num2str(iter),',Lambda:',num2str(lambda(l)), ...
                ',Alpha:',num2str(alpha(a)),',Performance:',num2str(performance{iter}(l,a).performance),...
                ',Interpretable:',num2str(interpretable_star{iter}(l,a).interpretability),',Plausible:',num2str(plausible_star{iter}(l,a))));
         end
    end
    save(strcat('Simulation_Results_shuffled.mat'),'performance','interpretable_star','plausible_star','interpretable_IBDS','plausible_IBDS','lambda','alpha','S1','S2','m1','m2');
end


%% Summarizing the results
for i = 1 : length(interpretable_star)
    for l = 1 : length(lambda)
        for a = 1 :length(alpha)
            perform(i,l,a)= performance{i}(l,a).performance;
            bias(i,l,a)= performance{i}(l,a).BS;
            ubVariance(i,l,a)= performance{i}(l,a).UV;
            bVariance(i,l,a)= performance{i}(l,a).BV;
            interpretability(i,l,a)= interpretable_star{i}(l,a).interpretability;
            representativeness(i,l,a)= interpretable_star{i}(l,a).representativeness;
            reproducibility(i,l,a)= interpretable_star{i}(l,a).reproducibility;
            variance(i,l,a)= ubVariance(i,l,a) - bVariance(i,l,a);
        end
    end
    plausibility(i,:,:) = plausible_star{i};
end

interpretabilityMean = squeeze(mean(interpretability,1));
interpretabilitySTD = squeeze(std(interpretability,[],1));
performanceMean = squeeze(mean(perform,1));
performanceSTD = squeeze(std(perform,[],1));
plausibilityMean = squeeze(mean(plausibility,1));
plausibilitySTD = squeeze(std(plausibility,[],1));
representativenessMean = squeeze(mean(representativeness,1));
representativenessSTD = squeeze(std(representativeness,[],1));
stabilityMean = squeeze(mean(reproducibility,1));
stabilitySTD = squeeze(std(reproducibility,[],1));
biasMean = squeeze(mean(bias,1));
biasSTD = squeeze(std(bias,[],1));
varianceMean = squeeze(mean(variance,1));
varianceSTD = squeeze(std(variance,[],1));

%% Plotting the data
figure;
scatter(x1(1,:),x1(2,:),20,'b.'),hold on, 
scatter(x2(1,:),x2(2,:),20,'r.'), hold on;
quiver(0,0,1,0,'g','LineWidth',2,'MaxHeadSize',1);
w = inv(S1)*X'*Y;
w = w/norm(w);
quiver(0,0,w(1),w(2),'m','LineWidth',2,'MaxHeadSize',1);
legend('Positive Samples','Negative Samples')
set(gca,'FontWeight','b')
