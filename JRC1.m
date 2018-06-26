function [del] = JRC1()%独立运行
%time: 程序运行的时间
%accurate 识别的正确率
%feature_num 每个图片的特征数
%clear;clc
global class_db train_num test_num class
%   声明全局变量
train_num = 4; %每个类训练图片的个数
test_num = 2; %每个类测试图片的个数
class_db = 50 ;  %人脸库中人脸类别的个数
feature_num = 150;
A = read_image;     %读取图片作为训练样本
% A = A6 ;
% class = [20] ;

prompt = '输入测试图片的类别，如[1,2,3]表示第1、2、3类图片\n';
class = input(prompt);
% test_class_num = 30;  %测试样本类别个数
% rand('seed',8);
% rand_class = randperm(class_db);%输入图片的类别,并且随机选取打乱的30个测试图片的顺序
% class = rand_class(1:test_class_num);
% class_db = length(class);  %类别个数
Y = read_image(class);      %读取测试样本
%%%%%%%%%%%%    PCA方法对图片降维
A = double(A');
A = zscore(A);  %中心化
[coeff,score] = pca(A);
A = score(:,1:feature_num);
A = A';
Y = double(Y');
Y = zscore(Y);  %中心化
Y = Y*coeff(:,1:feature_num);
Y = Y';%现在的A和Y都是降维后的
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% train_num = 4; %每个类训练图片的个数
% class_db = 25 ;  %人脸库中人脸类别的个数
% A = unifrnd(-20,20,60,100);
% Y = unifrnd(-20,20,60,30);
% 使用迭代方法求解稀疏矩阵X
t0 = clock;
% 参数设置
epsilon1 = 10^-5;
p = 1; 
lambda = 10;  %初始化lambda
sigma = 10^-10;
[m,d] = size(A);  %训练样本的像素和个数
n = size(Y,2);  %测试样本的个数
rand('seed',1);
X1 = rand(d,n);  %随机初始化一个X
% X1 = ones(d,n);
rho = 1;    %初始化一个rho，作为J1 和 J2的误差

%每张训练图片的类别指标
train_index = meshgrid(1:class_db ,1:train_num);
train_index = train_index(:);
train_index = train_index';
train.subindex = meshgrid(1:train_num,1:class_db);
train.subindex = train.subindex';
train_index(2,:) = train.subindex(:);
j=0;  % 用来记录一下迭代的步数
del = [];
W = eye(d);
while rho > epsilon1    
%     d = size(X1,1);
    X1_norm = arrayfun(@(i)norm(X1(i,:)),1:d); %计算X每一行的2范
    % 对X1 的行进行截断
%     X1_norm_s = find(X1_norm <= sigma); %返回行范数小于sigma的行指标
%     X1(X1_norm_s,:) = []; %去除这些行
%     A(:,X1_norm_s) =  []; %A 相应的列去掉
% %     记录下去掉的指标。
%     del = [del,train_index(:,X1_norm_s)];
%     train_index(:,X1_norm_s) = []; %更新指标,去掉相应的列
%     X1_norm(X1_norm_s) = [];
     %%%%%%%%%%%%%
%      权重
%     for i = 1:class_db
%        w(i) = sum(X1_norm((train_num * (i - 1) +1 ) : train_num * i)); 
%     end
%     W = meshgrid(w,1:train_num);
%     W = 1./W;
%     W = W(:);
%     W = diag(W); %生成对角阵
%     %%%%%

    H = diag(1./(X1_norm.^(2 - p)));   %计算H（k），对角元是A的i行2范数的倒数再2-p次幂的对角矩阵
%     H = W*H;%带权重测试行范数分布
    M = A'*A + 1/2*lambda*H;
    C = A'*Y;
    opts.POSDEF = true;opts.SYM = true;
    X2 = linsolve(M,C,opts);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    J1 = trace( ( A*X1 - Y )'*( A*X1 - Y ) ) + lambda*trace( X1'*H*X1 );
    J2 = trace( ( A*X2 - Y )'*( A*X2 - Y ) ) + lambda*trace( X2'*H*X2 );
    rho = 1 - J2/J1;
    X1 = X2;

end
%%%%%%%%    X已求出，下面即根据X进行图片的识别
X = X1;
%   最后截断
% X1_norm_s = find(X1_norm <= sigma); %返回行范数小于sigma的行指标
% del = train_index(:,X1_norm_s);
%%%%%%%%%%%%%%%%%%

% del = sortrows(del');      %输出一下删掉的列的类别
% del = del';
% uni_del = unique(del(1,:));
% l_del = length(uni_del);

% 对求出的X每一类的2范数画出来
d = size(X1,1);
xx = arrayfun(@(i)norm(X(i,:)),1:d); % X的每一行取2范数
figure(2)
bar(xx,'b')
xlabel('训练集图像')
ylabel('矩阵行取范数后的大小')
axis([0,d,0,max(xx)]);
% t1 = clock;
% time = etime(t1,t0);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% %根据X进行分类判断
% for i = 1:length(class)
%     T_Y = zeros(size(Y));
%     T_Y(:,test_num*(i-1) + 1 :test_num*i) = Y(:,test_num*(i-1) + 1 :test_num*i);
%     for j = 1:class_db
%        T_X = zeros(size(X)); 
%        T_X(train_num*(j-1)+1 : train_num*j,:) = X(train_num*(j-1)+1 : train_num*j,:);
%        error.matrix = T_Y - A*T_X;
%        norm_value(j) = norm( error.matrix(:,test_num*(i-1) + 1 :test_num*i),2 );
%     end
%     err(:,i) = norm_value;
%     min_norm(i) = min(norm_value); %范数值最小的那个
%     arg_min(i) = find(norm_value == min_norm(i)); 
% end
% time = etime(clock,t0)
% true1 = class == arg_min;    %正确为1，错误为0
% accurate = sum(true1)/length(true1)%正确率
end