function [true1,err] = JRC2( A,Y ,class) %返回重构误差
global class_db train_num test_num % 声明全局变量
% train_num = 4; %每个类训练图片的个数
% test_num = 2; %每个类测试图片的个数
% class_db = 50 ;  %人脸库中人脸类别的个数
% 使用迭代方法求解稀疏矩阵X
epsilon1 = 10^-5;
% lambda = 230;  %初始化lambda
lambda = 200;
rho = 1;    %初始化一个rho，作为J1 和 J2的误差
[m,d] = size(A);  %训练样本的像素和个数
n = size(Y,2);  %测试样本的个数
rand('seed',6);
X1 = rand(d,n);  %随机初始化一个X
while rho > epsilon1
    d = size(X1,1);
    X1_norm = arrayfun(@(i)norm(X1(i,:)),1:d); %计算X每一行的2范数
    H = diag(1./X1_norm);   %计算H（k），对角元是A的i行2范数的对觉矩阵
%     G = ones(m);    %因为这时候q= 2，G为单位矩阵，也可以不求出
    M = A'*A + 1/2*lambda*H;
    C = A'*Y;
    opts.POSDEF = true;
    opts.SYM = true;
    X2 = linsolve(M,C,opts);
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    J1 = trace( ( A*X1 - Y )'*( A*X1 - Y ) ) + lambda*trace( X1'*H*X1 );
    J2 = trace( ( A*X2 - Y )'*( A*X2 - Y ) ) + lambda*trace( X2'*H*X2 );
    rho = 1 - J2/J1;
    X1 = X2;
end
%%%%%%%%    X已求出，下面即根据X进行图片的识别
%%%%%%%%    这里暂且假设测试图片都在数据库中
X = X1;
for i = 1:length(class)
    T_Y = zeros(size(Y));
    T_Y(:,test_num*(i-1) + 1 :test_num*i) = Y(:,test_num*(i-1) + 1 :test_num*i);
    for j = 1:class_db
       T_X = zeros(size(X)); 
       T_X(train_num*(j-1)+1 : train_num*j,:) = X(train_num*(j-1)+1 : train_num*j,:);
       error.matrix = T_Y - A*T_X;
%        norm_value(j) = norm( error.matrix(:,test_num*(i-1) + 1 :test_num*i),'fro' );
       norm_value(j) = norm(error.matrix,'fro');
%        x1 = zeros(size(X,1),1);
%        x1(train_num*(j-1)+1 : train_num*j) = X(train_num*(j-1)+1 : train_num*j,i);
    end
    err(:,i) = norm_value;
    min_norm(i) = min(norm_value); %范数值最小的那个
    arg_min(i) = find(norm_value == min_norm(i));  
end

true1 = class == arg_min;    %正确为1，错误为0

end