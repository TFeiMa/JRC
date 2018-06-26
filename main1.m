%class = randperm(100);%输入图片的类别,并且随机选取打乱的30个测试图片的顺序

% 寻找JRC能够识别，而SRC不能识别的照片主程序
clear;clc
global class_db train_num test_num  % 声明全局变量
class_db = 100; %图片总类别（不同人数）个数
train_num = 5; %每个类训练图片的个数
test_num = 2; %每个类测试图片的个数
test_class_num = 90;  %测试样本类别个数

class = 1:90;
% A = read_image;     %读取图片作为训练样本
load('ATT.mat'); % 载入训练样本
load('FEI.mat');
A = FEI;
A = double(A');
A = zscore(A);  %中心化
[coeff,score,latent] = pca(A);
Y = read_image(class);
Y = double(Y');
Y = zscore(Y);  %中心化
A = score(:,1:200);
A = A';   
Y = Y*coeff(:,1:200);
Y = Y';
[true1,err1] = SRC2(A,Y,class);
[true2,err2] = JRC2(A,Y,class);
j = 1;
for i = 1:80
   if(true2(i) & ~(true1(2*i -1) & true1(2*i) ) )
       err(j) = i;
       j = j+1;
   end
end

j = 73;
err_j = err2(:,j);
err_s1 = err1(:,2*j -1);
err_s2 = err1(:,2*j);

plot(1:100,err_j,'o','MarkerFaceColor','b')
hold on
plot(1:100,err_s1,'*',1:100,err_s2,'+')
legend('JRC-联合图像','SRC-图像1','SRC-图像2')
y = find( err_s1 == min(err_s1));
hold on
plot(y,err_s1(y),'ko','markersize',13)
%hold on
%plot(1:100,err1(:,68),'*')
y = find(err_s2 == min(err_s2));
hold on
plot(y,err_s2(y),'bo','markersize',13)
hold on
%plot(1:100,err2(:,34),'v')
y = err_j;
%hold on 
plot(j,y(j),'ko','markersize',13)
xlabel('人脸类别');
ylabel('误差');
