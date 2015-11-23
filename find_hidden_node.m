function find_hidden_node(start_index , end_index , input_sequence ,day_number, node_number)
%find the most fittabel hidden layer node number

H_num = start_index;
Force_train = 1;
node_number.hidden = H_num;
if end_index < start_index
    error('end_index must be lagger than start_index');
    return;
end
legend_label = [];
line_style = ['b-+';'k-.';'r-*';'k-o';'r--';'k-x'];
figure(6);
hold on;
for H_num=start_index:1:end_index
    temp_label = ['隐含层节点数为' num2str(H_num)];
    legend_label = [legend_label ; temp_label];
    node_number.hidden = H_num;
    Wave_params = wave_nerve(input_sequence ,day_number, node_number ,Force_train);
    plot(Wave_params.errors(1,1:5:end) , line_style(H_num-start_index+1,:),'markersize',4);
end
xlabel('迭代次数','FontSize',15);
ylabel('误差','FontSize',15);
title('不同隐含层节点数下神经网络模型的预测误差曲线','FontSize',15);
legend(legend_label(:,:));
axis tight;
