function [accurate] = JRC_Classify(X,A,Y)
% %根据X进行分类判断
% Y是测试矩阵，A是训练矩阵
global class_db train_num test_num class %全局变量

for i = 1:length(class)
    T_Y = zeros(size(Y));
    T_Y(:,test_num*(i-1) + 1 :test_num*i) = Y(:,test_num*(i-1) + 1 :test_num*i);
    for j = 1:class_db
       T_X = zeros(size(X)); 
       T_X(train_num*(j-1)+1 : train_num*j,:) = X(train_num*(j-1)+1 : train_num*j,:);
       error.matrix = T_Y - A*T_X;
       norm_value(j) = norm( error.matrix,2 );
    end
    min_norm = min(norm_value); %范数值最小的那个
    arg_min(i) = find(norm_value == min_norm); 
%     arg_min(i) = temp(1);
end
% time = etime(clock,t0)
true1 = class == arg_min;    %正确为1，错误为0
accurate = sum(true1)/length(true1)%正确率

end