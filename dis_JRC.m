function[time,accurate] = dis_JRC( A,Y ,class,l)
% 当测试样本的类别很多时，计算的时间开销将很大，考虑将测试样本类别
% 分开表示、识别
global test_num
class_num = length(class);
Y1 = Y(:,1:class_num*test_num/2);
Y2 = Y(:,(class_num*test_num/2+1):end);
class1 = class(1:class_num/2);
class2 = class((class_num/2+1):end);
AA = A;
lambda = l ;
[time1, accurate_num1] = JRC(AA,Y1,class1,lambda);
[time2,accurate_num2] = JRC(AA,Y2,class2,lambda);
time = time1 + time2;
accurate = (accurate_num1 + accurate_num2)/(class_num);
end