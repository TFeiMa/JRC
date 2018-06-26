function [time,accurate] = pca_test(A,Y,class )
%使用主成分分析的方法进行人脸识别
%每个类别的训练样本只选取5个图片
%time: 程序运行的时间
%accurate 识别的正确率
%feature_num 每个图片的特征数
%clear;clc;
% train_num = 5; %每个类训练图片的个数
% test_num = 2; %每个类测试图片的个数
% class_db = 100 ;  %人脸库中人脸类别的个数
global class_db train_num test_num 
% 声明全局变量

t0 = clock;
if 0
A = read_image;     %读取图片作为训练样本
%prompt = '输入测试图片的类别（1—40），如[1,2,3]表示第1、2、3类图片\n';
%class = input(prompt);
%class_db = [2];
Y = read_image(class);      %读取测试样本
%tic;%计时开始
t0 = clock;
%%%%    downsample下采样
%if 0
A = downsample(double(A),100);
Y = downsample(double(Y),100);
%end
%%%%

%%%%%%%%%%%%    PCA方法对图片降维
A = double(A');
A = zscore(A);  %中心化
[coeff,score,latent] = pca(A);
l = cumsum(latent)/sum(latent);
p_num = sum(l<=0.95);%设定一个阈值0.95，计算主成分的个数
if nargin == 0
    feature_num = p_num;
end
A = score(:,1:feature_num);
A = A';
Y = double(Y');
Y = zscore(Y);  %中心化
Y = Y*coeff(:,1:feature_num);
Y = Y';
end
%现在的A和Y都是降维后的
%使用主成分分析，这个时候的A的列表示训练样本专用每个类的稀疏
%Y表示测试样本对主成分的系数
cla = meshgrid(class,1:test_num);
cla = cla(:);   %所有测试照片的类别
all_test = size(Y,2);%所有测试图像的个数
%下面基于距离来进行判别
%计算测试图片与train_num个图片的距离和
for i = 1:all_test
   y = Y(:,i);
   for j = 1:class_db
       count = 0;
       for k = 1:train_num
           count = count +norm(y - A(:,train_num*(j-1)+k));
       end
       e(j) = count;
   end
   arg_min(i,:) = find(e == min(e)); %识别的类别
end
true = cla == arg_min;
accurate = sum(true)/all_test;  %准确率
%toc;%计时结束
%t1=clock;
time = etime(clock,t0);
end


