function [ A ] = read_image( class )
%function [ A ] = read_image( class,test_num )
global class_db train_num test_num   % 声明全局变量
%   class_db 想要读取照片的类别，默认读取整个人脸库作为训练集
%   返回A作为读取图片的矩阵，其中A的每一列作为一个样本（图片）
% class_db = 50; %图片总类别（不同人数）个数
% train_num = 4; %每个类训练图片的个数
% test_num = 2; %每个类测试图片的个数

if nargin == 0      %默认参数，默认情况下读取训练图片
    class = 1:class_db;
    each_class = 2:2:2*train_num;   %每个att_face类(人)读取的图片，默认读前七张
    %each_class = [110];
elseif nargin ~= 0
    %each_class = (train_num+1) : (train_num+test_num);  %需要att_face取的测试图片
    each_class = 1:2:2*test_num; 
%     each_class = [1,10];
end

A_col = 0; %用来计算矩阵A的列数

for i = class
    
    for j = each_class
         %att_face database
        file_path =fullfile('F:\face database\att_faces',strcat('s',int2str(i)),int2str(j));   
        temp = imread(strcat(file_path,'.pgm'));
        
        %%%%%%%%%%%%%%%%%   
%         FEI face database

%         if j < 10
%             file_path = fullfile('F:\face database\FEI Face Database\images',...
%                 strcat(int2str(i),'-0',int2str(j)));
%         else
%             file_path = fullfile('F:\face database\FEI Face Database\images',...
%                 strcat(int2str(i),'-',int2str(j)));
%         end
%         temp = imread(strcat(file_path,'.jpg'));
%         temp = imresize(temp,[480,640]);    %把图片统一大小
%         temp = rgb2gray(temp);  %RGB 图像转化为 gray图像
%         temp = temp(1:10:480,1:10:640);%下采样
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        [m,n] = size(temp);
        A(:,A_col + 1 ) =  reshape(temp',[m*n,1]);
        A_col = size(A,2);  %更新A的列数
    end
end

end