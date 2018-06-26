% function [time,accurate] = SRC( A,Y ,class ,l)
% function [time,accurate] = SRC( A,Y ,class)
function [true,err] = SRC( A,Y ,class)
%time: 程序运行的时间
%accurate 识别的正确率
%feature_num 每个图片的特征数
%clear;clc;
global class_db train_num test_num  % 声明全局变量
% train_num = 5; %每个类训练图片的个数
% test_num = 2; %每个类测试图片的个数
% class_db = 100 ;  %人脸库中人脸类别的个数
t0 = clock; %初始时间
cla = meshgrid(class,1:test_num);
cla = cla(:);   %所有测试照片的类别
all_test = size(Y,2);%所有测试图像的个数

for i = 1:all_test
   y = Y(:,i);
%    x = lasso(A,y,'lambda',0.1);
   x = lasso(A,y,'lambda',l);
   for j = 1:class_db
       x1 = zeros(size(x));
       x1(train_num*(j-1)+1 : train_num*j) = x(train_num*(j-1)+1 : train_num*j);
       y1(j) = norm(y - A*x1);
   end
%    err(:,i) = y1;
    m_y1 = find(y1 == min(y1));
    if(length(m_y1)<2)
        arg_min(i,:) = m_y1; %识别的类别
    else
       arg_min(i,:) =  m_y1(1);
    end
   
end
true = cla == arg_min;
accurate = sum(true)/all_test;  %准确率
%toc;%计时结束
time = etime(clock,t0);
%end
end