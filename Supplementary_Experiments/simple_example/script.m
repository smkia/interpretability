addpath(genpath('\Functions'))
% Data without Noise
sample_num = 10000;
[X, Y] = linear_data([sample_num,sample_num],1,0,0);
figure,scatter(X(1:sample_num,1),X(1:sample_num,2),5,'r.'), hold on, scatter(X(sample_num+1:end,1),X(sample_num+1:end,2),5,'b.')
legend('Negative Class, y=-1', 'Positive Class, y=1')
xlabel('$\textit{x}_1$','interpreter','latex')
ylabel('$\textit{x}_2$','interpreter','latex')
quiver(0.5,0.5,-0.2,0.2,'k','LineWidth',2,'MaxHeadSize',1);
set(gca,'FontWeight','b')

% Data with noise
sample_num = 10000;
[X, Y] = linear_data([sample_num,sample_num],1,0,1,[0.02,-0.01;-0.01,1]);
figure,scatter(X(1:sample_num,1),X(1:sample_num,2),5,'r.'), hold on, scatter(X(sample_num+1:end,1),X(sample_num+1:end,2),5,'b.')
legend('Negative Class, y=-1', 'Positive Class, y=1')
xlabel('$\textit{x}_1$','interpreter','latex')
ylabel('$\textit{x}_2$','interpreter','latex')
xlim([-0.5,1.5])
ylim([-5,5])
X = mapstd(X')';
W = X'*X\X'*Y;
quiver(0.5,0.2,W(1)/2,W(2)/2,'g','LineWidth',2,'MaxHeadSize',5);
set(gca,'FontWeight','b')


iter = 100;
W_star = [-1;1];
for i = 1 : iter
    [X, Y] = linear_data([sample_num,sample_num],1,0,1,[0.02,-0.01;-0.01,1]);
    X = mapstd(X')';
    W = X'*X\X'*Y;
    y_pred = sign(X*W_star);
    W_star_acc(i) = mean(Y==y_pred);
    y_pred = sign(X*W);
    W_acc(i) = mean(Y==y_pred);
    disp(i);
end
