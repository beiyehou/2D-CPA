function y_out=change_amp(x_input , amp)
%等比例修改数组的最大最小值
[rows , cols] = size(x_input);
for i=1:1:rows
    x_input(i,:)=amp*norm_change(x_input(i,:));
end

y_out = x_input;