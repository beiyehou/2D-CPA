function Arima_params = train_arima( data )
% 本函数用于训练得出 arima 模型的各个参数

% data 输入数据
% Arima_params 输出数据 (结构体)

source_data = dimension_change(data,'row');

% 计算一次趋势项
% source_data = detrend( source_data );
% trend_data = source_data - data;
% fit_p = polyfit( [1:length(trend_data)] , trend_data , 1 );

% 计算差分次数
H = adftest( source_data );
difftime = 0;
temp_data = source_data;
while ~H
    temp_data = diff( temp_data );
    difftime = difftime + 1;
    H = adftest( temp_data );
end

% 计算 p q 值
temp_data = dimension_change(temp_data , 'col');
u = iddata(temp_data);
check = [];

for p = 1:4          
    for q = 1:4                  
        m = armax(u,[p q]);        
        AIC = aic(m);              
        check = [check;p q AIC];
    end
end
[best_value , best_index]= min(check(:,3));
p_best = check( best_index , 1 );
q_best = check( best_index , 2 );

% 参数返回
Arima_params.I = difftime;
Arima_params.p = p_best;
Arima_params.q = q_best;
Arima_params.aic = best_value;
% Arima_params.fit_p = fit_p;
Arima_params.fit_index = length( data );
Arima_params.histroy_iddata = u;