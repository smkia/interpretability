clear all;
n = 10000;
p = [5,10,50,100,500,1000,5000,10000];
figure;
for j = 1 : length(p)
    refrence_vector = random('Uniform',-1,1,1,p(j));
    X = random('Uniform',-1,1,n,p(j));
    for i = 1 : n
        d(i,j) = 1 - pdist([refrence_vector;X(i,:)],'cosine');
    end
    subplot(2,4,j)
    [mu(j),sig(j)]=normfit(d(:,j));
    sighat(j) = sqrt(1./p(j));
    histfit(d(:,j),100,'normal');
    title(strcat('p = ',num2str(p(j))),'FontWeight','bold');
    set(gca,'LineWidth',2,'FontWeight','bold');
    critVal(j) = prctile(abs(d(:,j)),95);
    [h(j),p_value(j)] = adtest(d(:,j));
    disp(j)
end