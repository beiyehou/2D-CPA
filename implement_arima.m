function prediction_result = implement_arima( Arima_params , data)
% ����������ʵʩ arima �㷨

% Arima_params ����ģ�ͽṹ��
% data �����Ԥ������
% prediction_result ���Ԥ����

data = dimension_change(data,'row');
% data = data(1,end-(Arima_params.p+Arima_params.I+10):end);
% ����Ƿ���Ҫ���
temp_diff = Arima_params.I;
savediffdata = [];
predict_data = data;
if temp_diff > 0
    for i=1:1:temp_diff
        savediffdata = [ savediffdata predict_data(1,1) ];
        temp_diff = temp_diff - 1;
        predict_data = diff( predict_data );
    end
end

% ʵʩ arima Ԥ��
 predict_data = dimension_change(predict_data,'col');
 predict_iddata = iddata( predict_data );

%  histroy_iddata =iddata( predict_iddata );

%  model = armax( histroy_iddata , [Arima_params.p  Arima_params.q] );
%  present(model);
 predict_object =  predict(Arima_params.model , predict_iddata , 1 );
 
 model_predict_data = predict_object.OutputData'; % ��ת��תΪ������
  
 if size(savediffdata,2) ~= 0
      model_predict_data = [predict_data(1,1:1:end-1) , model_predict_data(1,end)];
      for index=size(savediffdata,2):-1:1
            model_predict_data=cumsum([savediffdata(index),model_predict_data]);   
      end
end 

 
 % ����Ԥ����
 prediction_result = model_predict_data;
 
 

