clear;clc;
% 定义一组全局变量
global class_db train_num test_num  test_class_num d_min d_max 
class_db = 100; %图片总类别（不同人数）个数
train_num = 5; %每个类训练图片的个数
test_num = 3; %每个类测试图片的个数
test_class_num = 80;  %测试样本类别个数
d_min = 10;
d_max = 250; %维度的最大值
load('ATT.mat');% 载入训练样本
load('FEI.mat');
A = FEI;
rand('seed',8);
rand_class = randperm(class_db);%输入图片的类别,并且随机选取打乱的30个测试图片的顺序
class = rand_class(1:test_class_num);
dimision =  d_min :30 : d_max;

lambda = [0.001,0.01,0.1,1,10,100,300,1000] ;
dim = 200;

% A = read_image;     %读取图片作为训练样本
%%%%%%%%%%%%    PCA方法对图片降维
A = double(A');
A = zscore(A);  %中心化
[coeff,score,latent] = pca(A);
Y = read_image(class);      %读取测试样本
Y = double(Y');
Y = zscore(Y);  %中心化
j = 1;
for dim = dimision
% for l = lambda
    A1 = score(:,1:dim);
    A1 = A1';
    Y1 = Y*coeff(:,1:dim);
    Y1 = Y1';
    %现在的A和Y都是降维后的    
%     [jrc_time(j),jrc_accurate(j)] = JRC(A1,Y1,class);
%     [jrc_time(j),jrc_accurate(j)] = dis_JRC(A1,Y1,class);
%     [src_time(j),src_accurate(j)] = SRC(A1,Y1,class);
%     [pca_time(j),pca_accurate(j)] = pca_test(A1,Y1,class);
%     [svm_time(j),svm_accurate(j)] = svm(A1,Y1,class);

    [jrc_time(j,:),jrc_accurate(j,:)] = dis_JRC(A1,Y1,class,l);
    [src_time(j,:),src_accurate(j,:)] = SRC(A1,Y1,class,l);
   
    j=j+1;
end  
lambda = lambda';
figure(1)
semilogx(lambda,jrc_time,'-*')
hold on
semilogx(lambda,src_time,'-o')

% plot(dimision,jrc_time,'-*')
% hold on
% plot(dimision,src_time,'-o')
% hold on 
% plot(dimision,pca_time,'-^')
% hold on 
% plot(dimision,svm_time,'-+')
% title('时间对比图');
% xlabel('lambda');
% ylabel('时间(s)');
% legend('JRC','SRC','PCA','SVM');
% axis([d_min,d_max,-1,50]);
legend('JRC','SRC');

figure(2)
semilogx(lambda,100*jrc_accurate,'-*')
hold on
semilogx(lambda,100*src_accurate,'-o')


% plot(dimision,100*jrc_accurate,'-*')
% hold on
% plot(dimision,100*src_accurate,'-o')
% hold on
% plot(dimision,100*pca_accurate,'-^')
% hold on
% plot(dimision,svm_accurate,'-+')
% % title('识别率与lambda关系');
% title('识别率对比图');
% xlabel('特征维度');
% ylabel('识别率(%)');
% legend('JRC','SRC','PCA','SVM');
% axis([d_min,d_max,0,100]);
legend('JRC','SRC');
