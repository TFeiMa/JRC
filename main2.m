clear;clc;
% 平均情况下结果
global class_db train_num test_num  test_class_num d_min d_max 
class_db = 100; %图片总类别（不同人数）个数
train_num = 5; %每个类训练图片的个数
test_num = 3; %每个类测试图片的个数
test_class_num = 80;  %测试样本类别个数
d_min = 10;  %维度的最小值
d_max = 300; %维度的最大值
load('ATT.mat'); % 载入训练样本
load('FEI.mat');
A = FEI;   

dimision =  d_min : 30 : d_max;
% A = read_image;     %读取图片作为训练样本
%%%%%%%%%%%%    PCA方法对图片降维
A = double(A');
A = zscore(A);  %中心化
[coeff,score,latent] = pca(A);
for j = 1:5
    i = 1;
    rand_class = randperm(class_db);%输入图片的类别,并且随机选取打乱的30个测试图片的顺序
    class = rand_class(1:test_class_num);
    Y = read_image(class);      %读取测试样本
    Y = double(Y');
    Y = zscore(Y);  %中心化
    for dim = dimision
        A1 = score(:,1:dim);
        A1 = A1';
        Y1 = Y*coeff(:,1:dim);
        Y1 = Y1';
    %现在的A和Y都是降维后的    
%        [jrc_time(i,j),jrc_accurate(i,j)] = JRC(A1,Y1,class);
       [jrc_time(i,j),jrc_accurate(i,j)] = dis_JRC(A1,Y1,class);
       [src_time(i,j),src_accurate(i,j)] = SRC(A1,Y1,class);
       [pca_time(i,j),pca_accurate(i,j)] = pca_test(A1,Y1,class);
       [svm_time(i,j),svm_accurate(i,j)] = svm(A1,Y1,class);
       i= i+1;
    end  
end
mjrc_time = mean(jrc_time,2);mjrc_accurate = mean(jrc_accurate,2);
msrc_time = mean(src_time,2);msrc_accurate = mean(src_accurate,2);
mpca_time = mean(pca_time,2);mpca_accurate = mean(pca_accurate,2);
msvm_time = mean(svm_time,2);msvm_accurate = mean(svm_accurate,2);
% 
% vjrc_time = var(jrc_time,2);vjrc_accurate = var(jrc_accurate,2);
% vsrc_time = var(src_time,2);vsrc_accurate = var(src_accurate,2);
% vpca_time = var(pca_time,2);vpca_accurate = var(pca_accurate,2);
% vsvm_time = var(svm_time,2);vsvm_accurate = var(svm_accurate,2);
% 
dimision = dimision';
figure(1)
% subplot(2,2,3);
plot(dimision,mjrc_time,'-*')
hold on
plot(dimision,msrc_time,'-o')
hold on 
plot(dimision,mpca_time,'-^')
hold on 
plot(dimision,msvm_time,'-+')
title('时间对比图');
xlabel('图像特征维数');
ylabel('时间(s)');
legend('JRC','SRC','PCA','SVM');
axis([d_min,d_max,-1,50]);

% subplot(2,2,4);
figure(2)
plot(dimision,100*mjrc_accurate,'-*')
hold on
plot(dimision,100*msrc_accurate,'-o')
hold on
plot(dimision,100*mpca_accurate,'-^')
hold on
plot(dimision,msvm_accurate,'-+')
title('识别率对比图');
xlabel('图像特征维数');
ylabel('识别率(%)');
legend('JRC','SRC','PCA','SVM');
axis([d_min,d_max,0,100]);