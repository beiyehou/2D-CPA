function varargout = traffic(varargin)
% TRAFFIC MATLAB code for traffic.fig
%      TRAFFIC, by itself, creates a new TRAFFIC or raises the existing
%      singleton*.
%
%      H = TRAFFIC returns the handle to a new TRAFFIC or the handle to
%      the existing singleton*.
%
%      TRAFFIC('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in TRAFFIC.M with the given input arguments.
%
%      TRAFFIC('Property','Value',...) creates a new TRAFFIC or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before traffic_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to traffic_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help traffic

% Last Modified by GUIDE v2.5 23-Apr-2015 08:30:41

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @traffic_OpeningFcn, ...
                   'gui_OutputFcn',  @traffic_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT

% 定义最大流量值 单位 Mbit/s
global traffic_max;
traffic_max = 720;

% --- Executes just before traffic is made visible.
function traffic_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to traffic (see VARARGIN)

% Choose default command line output for traffic
handles.output = hObject;


% 定义显示在 text_show_result 的结构体 和 存储流量数据的 数组 
text_show_result.room = '1x';
text_show_result.range = 0.0;
text_show_result.show_global = 'Y';
text_show_result.show_detail = 'N';
text_show_result.slider = 0.0;
handles.show_result = text_show_result;
try 
    load('saved/preducer.mat','traffic_data','man_data');
catch
    traffic_data = zeros(1,24*60/5);
    man_data = [];
end
handles.traffic_data = traffic_data;
handles.man_data = man_data;
handles.man_data_length = size(man_data,1);
% Update handles structure
guidata(hObject, handles);

% UIWAIT makes traffic wait for user response (see UIRESUME)
% uiwait(handles.global_figure);


% --- Outputs from this function are returned to the command line.
function varargout = traffic_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on slider movement.
function slider0_Callback(hObject, eventdata, handles)
% hObject    handle to slider0 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
draw_show_result(hObject,handles);
draw_figure(hObject,handles);

% --- Executes during object creation, after setting all properties.
function slider0_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider0 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
slider_handler_remember(hObject,handles);

% --- Executes on slider movement.
function slider1_Callback(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
draw_show_result(hObject,handles);
draw_figure(hObject,handles);
% --- Executes during object creation, after setting all properties.
function slider1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

slider_handler_remember(hObject,handles);
% --- Executes on slider movement.
function slider2_Callback(hObject, eventdata, handles)
% hObject    handle to slider2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
draw_show_result(hObject,handles);
draw_figure(hObject,handles);
% --- Executes during object creation, after setting all properties.
function slider2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
slider_handler_remember(hObject,handles);

% --- Executes on slider movement.
function slider3_Callback(hObject, eventdata, handles)
% hObject    handle to slider3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
draw_show_result(hObject,handles);
draw_figure(hObject,handles);
% --- Executes during object creation, after setting all properties.
function slider3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

slider_handler_remember(hObject,handles);
% --- Executes on slider movement.
function slider4_Callback(hObject, eventdata, handles)
% hObject    handle to slider4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
draw_show_result(hObject,handles);
draw_figure(hObject,handles);
% --- Executes during object creation, after setting all properties.
function slider4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
slider_handler_remember(hObject,handles);

% --- Executes on slider movement.
function slider5_Callback(hObject, eventdata, handles)
% hObject    handle to slider5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
draw_show_result(hObject,handles);
draw_figure(hObject,handles);
% --- Executes during object creation, after setting all properties.
function slider5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
slider_handler_remember(hObject,handles);

% --- Executes on slider movement.
function slider6_Callback(hObject, eventdata, handles)
% hObject    handle to slider6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
draw_show_result(hObject,handles);
draw_figure(hObject,handles);
% --- Executes during object creation, after setting all properties.
function slider6_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

slider_handler_remember(hObject,handles);
% --- Executes on slider movement.
function slider7_Callback(hObject, eventdata, handles)
% hObject    handle to slider7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
draw_show_result(hObject,handles);
draw_figure(hObject,handles);
% --- Executes during object creation, after setting all properties.
function slider7_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
slider_handler_remember(hObject,handles);

% --- Executes on slider movement.
function slider8_Callback(hObject, eventdata, handles)
% hObject    handle to slider8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
draw_show_result(hObject,handles);
draw_figure(hObject,handles);
% --- Executes during object creation, after setting all properties.
function slider8_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
slider_handler_remember(hObject,handles);

% --- Executes on slider movement.
function slider9_Callback(hObject, eventdata, handles)
% hObject    handle to slider9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
draw_show_result(hObject,handles);
draw_figure(hObject,handles);
% --- Executes during object creation, after setting all properties.
function slider9_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

slider_handler_remember(hObject,handles);
% --- Executes on slider movement.
function slider10_Callback(hObject, eventdata, handles)
% hObject    handle to slider10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
draw_show_result(hObject,handles);
draw_figure(hObject,handles);
% --- Executes during object creation, after setting all properties.
function slider10_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
slider_handler_remember(hObject,handles);

% --- Executes on slider movement.
function slider11_Callback(hObject, eventdata, handles)
% hObject    handle to slider11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
draw_show_result(hObject,handles);
draw_figure(hObject,handles);
% --- Executes during object creation, after setting all properties.
function slider11_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
slider_handler_remember(hObject,handles);

% --- Executes on slider movement.
function slider12_Callback(hObject, eventdata, handles)
% hObject    handle to slider12 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
draw_show_result(hObject,handles);
draw_figure(hObject,handles);
% --- Executes during object creation, after setting all propertiedraw_figure(hObject,handles);s.
function slider12_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider12 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

slider_handler_remember(hObject,handles);
% --- Executes on slider movement.
function slider13_Callback(hObject, eventdata, handles)
% hObject    handle to slider13 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
draw_show_result(hObject,handles);
draw_figure(hObject,handles);
% --- Executes during object creation, after setting all properties.
function slider13_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider13 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
slider_handler_remember(hObject,handles);

% --- Executes on slider movement.
function slider14_Callback(hObject, eventdata, handles)
% hObject    handle to slider14 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
draw_show_result(hObject,handles);
draw_figure(hObject,handles);
% --- Executes during object creation, after setting all properties.
function slider14_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider14 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

slider_handler_remember(hObject,handles);
% --- Executes on slider movement.
function slider15_Callback(hObject, eventdata, handles)
% hObject    handle to slider15 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
draw_show_result(hObject,handles);
draw_figure(hObject,handles);
% --- Executes during object creation, after setting all properties.
function slider15_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider15 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
slider_handler_remember(hObject,handles);

% --- Executes on slider movement.
function slider16_Callback(hObject, eventdata, handles)
% hObject    handle to slider16 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
draw_show_result(hObject,handles);
draw_figure(hObject,handles);
% --- Executes during object creation, after setting all properties.
function slider16_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider16 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
slider_handler_remember(hObject,handles);

% --- Executes on slider movement.
function slider17_Callback(hObject, eventdata, handles)
% hObject    handle to slider17 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
draw_show_result(hObject,handles);
draw_figure(hObject,handles);
% --- Executes during object creation, after setting all properties.
function slider17_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider17 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
slider_handler_remember(hObject,handles);

% --- Executes on slider movement.
function slider18_Callback(hObject, eventdata, handles)
% hObject    handle to slider18 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
draw_show_result(hObject,handles);
draw_figure(hObject,handles);
% --- Executes during object creation, after setting all properties.
function slider18_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider18 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
slider_handler_remember(hObject,handles);

% --- Executes on slider movement.
function slider19_Callback(hObject, eventdata, handles)
% hObject    handle to slider19 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
draw_show_result(hObject,handles);
draw_figure(hObject,handles);
% --- Executes during object creation, after setting all properties.
function slider19_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider19 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
slider_handler_remember(hObject,handles);

% --- Executes on slider movement.
function slider20_Callback(hObject, eventdata, handles)
% hObject    handle to slider20 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
draw_show_result(hObject,handles);
draw_figure(hObject,handles);
% --- Executes during object creation, after setting all properties.
function slider20_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider20 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
slider_handler_remember(hObject,handles);

% --- Executes on slider movement.
function slider21_Callback(hObject, eventdata, handles)
% hObject    handle to slider21 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
draw_show_result(hObject,handles);
draw_figure(hObject,handles);
% --- Executes during object creation, after setting all properties.
function slider21_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider21 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
slider_handler_remember(hObject,handles);

% --- Executes on slider movement.
function slider22_Callback(hObject, eventdata, handles)
% hObject    handle to slider22 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
draw_show_result(hObject,handles);
draw_figure(hObject,handles);
% --- Executes during object creation, after setting all properties.
function slider22_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider22 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

slider_handler_remember(hObject,handles);
% --- Executes on slider movement.
function slider23_Callback(hObject, eventdata, handles)
% hObject    handle to slider23 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
draw_show_result(hObject,handles);
draw_figure(hObject,handles);
% --- Executes during object creation, after setting all properties.
function slider23_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider23 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
slider_handler_remember(hObject,handles);


% --- Executes on slider movement.
function slider_range_Callback(hObject, eventdata, handles)
% hObject    handle to slider_range (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
[hObject,handles] = draw_show_result(hObject,handles);
set_slider_value(hObject,handles);
% --- Executes during object creation, after setting all properties.
function slider_range_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider_range (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

% --- Executes on button press in checkbox_global.
function checkbox_global_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_global (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox_global
draw_show_result(hObject,handles);

% --- Executes on button press in checkbox_detail.
function checkbox_detail_Callback(hObject, eventdata, handles)
% hObject    handle to checkbox_detail (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkbox_detail
draw_show_result(hObject,handles);

% --- Executes on button press in pushbutton_reset.
function pushbutton_reset_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_reset (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

draw_figure(hObject,handles);

% --- Executes on button press in pushbutton_exit.
function pushbutton_exit_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_exit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

delete(handles.global_figure);
clear all;
% --- Executes on button press in pushbutton_save.
function pushbutton_save_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_save (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
disp(handles.traffic_data);
traffic_data = handles.traffic_data;
man_data = handles.man_data;
man_data(size(man_data,1)+1,:) = traffic_data;
handles.man_data = man_data;
handles.man_data_length = size(man_data,1);
save('saved/preducer.mat','traffic_data','man_data');
guidata(hObject, handles);
% --- Executes on scroll wheel click while the figure is in focus.
function global_figure_WindowScrollWheelFcn(hObject, eventdata, handles)
% hObject    handle to global_figure (see GCBO)
% eventdata  structure with the following fields (see FIGURE)
%	VerticalScrollCount: signed integer indicating direction and number of clicks
%	VerticalScrollAmount: number of lines scrolled for each click
% handles    structure with handles and user data (see GUIDATA)
% --- Executes during object creation, after setting all properties.


% --- Executes when selected object is changed in uipanel_room.
function uipanel_room_SelectionChangeFcn(hObject, eventdata, handles)
% hObject    handle to the selected object in uipanel_room 
% eventdata  structure with the following fields (see UIBUTTONGROUP)
%	EventName: string 'SelectionChanged' (read only)
%	OldValue: handle of the previously selected object or empty if none was selected
%	NewValue: handle of the currently selected object
% handles    structure with handles and user data (see GUIDATA)
[hObject,handles] = draw_show_result(hObject,handles);
set_slider_value(hObject,handles);



function [hObject,handles] = draw_show_result(hObject,handles)
tag_slider = '';
tag_string = get(hObject,'tag');
switch tag_string
    case 'radiobutton_1x'
        handles.show_result.room = get(hObject,'String');
    case 'radiobutton_2x'
        handles.show_result.room = get(hObject,'String');
    case 'checkbox_global'
        handles.show_result.show_global = comma(get(hObject,'Value') == 1,'Y','N');
    case 'checkbox_detail'
        handles.show_result.show_detail = comma(get(hObject,'Value') == 1,'Y','N');
    case 'slider_range'
        handles.show_result.range = get(hObject,'Value');
    otherwise
        handles.show_result.slider = get(hObject,'Value');
        tag_slider = tag_string;
end

set(handles.text_show_result,'String',['Room:' handles.show_result.room ' | ' 'Show:' 'global-' handles.show_result.show_global  'detail-' handles.show_result.show_detail ' | ' ...
    'Range:' num2str(floor(handles.show_result.range)) '-' num2str(ceil(handles.show_result.range)) ' | ' ...
    'Slider:' tag_slider ' | '  'Value:'  num2str(handles.show_result.slider) ' | ' '第' num2str(handles.man_data_length) '天']);

guidata(hObject, handles);

function draw_figure(hObject,handles)
global traffic_max;
tag_string = get(hObject,'tag');
tag_style = get(hObject,'Style');

switch tag_style
    case 'slider'
        if strcmp(handles.show_result.room , '1x')
            handles.traffic_data(1,str2num(tag_string(length(tag_style)+1:end))*60/5+1) = get(hObject,'Value');
        elseif strcmp(handles.show_result.room , '2x')
            if floor(handles.show_result.range)*60/5 + str2num(tag_string(length(tag_style)+1:end)) +1  <= 24*60/5
                handles.traffic_data(1, floor(handles.show_result.range)*60/5 ...
                    + str2num(tag_string(length(tag_style)+1:end)) +1 ) = get(hObject,'Value');
            end
        end
    case 'pushbutton'
        if strcmp(tag_string ,'pushbutton_reset')
            try 
                load('saved/preducer.mat','traffic_data','man_data');
                handles.traffic_data = man_data(22,:);
            catch
                traffic_data = zeros(1,24*60/5);
                handles.traffic_data = traffic_data;
            end
            
        end
end
guidata(hObject, handles);

axes(handles.axes_figure); 
% 生成 24 小时制 xlabel
xlabel_str = {'0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24'};    

area(handles.traffic_data,'FaceColor',[0.0,0.6,0.0]);
set(handles.axes_figure,'xtick', [1:60/5:24*60/5]);
set(handles.axes_figure,'xticklabel' , xlabel_str);
set(handles.axes_figure,'xticklabel' , xlabel_str);
axis([1 24*60/5 0 traffic_max]);
grid on;

function set_slider_value(hObject,handles)
global slider_handler ;
tag_string = get(hObject,'tag');
tag_style = get(hObject,'Style');
switch tag_style
    case 'radiobutton'
            if strcmp(get(hObject,'String') , '1x')
                for i=0:1:23
                    set(slider_handler(1,i+1),'Value',handles.traffic_data(1,i*60/5+1));
                end
            elseif strcmp(get(hObject,'String') , '2x')
                for i=0:1:23
                     if floor(handles.show_result.range)*60/5 + i +1  <= 24*60/5
                        set(slider_handler(1,i+1),'Value',handles.traffic_data(1,floor(handles.show_result.range)*60/5 + i +1));
                     end
                end
            end        
    case 'slider'
            if strcmp(handles.show_result.room , '1x')
                for i=0:1:23
                    set(slider_handler(1,i+1),'Value',handles.traffic_data(1,i*60/5+1));
                end
            elseif strcmp(handles.show_result.room , '2x')
                for i=0:1:23
                     if floor(get(hObject,'Value'))*60/5 + i +1  <= 24*60/5
                        set(slider_handler(1,i+1),'Value',handles.traffic_data(1,floor(get(hObject,'Value'))*60/5 + i +1));
                     end
                end
            end          
end

guidata(hObject, handles);

function slider_handler_remember(hObject,handles)
tag_string = get(hObject,'tag');
tag_style = get(hObject,'Style');
global slider_handler ;
global traffic_max;
if strcmp(tag_style,'slider')
    slider_handler(1,str2num(tag_string(length(tag_style)+1:end))+1) = hObject;
    set(hObject,'Max',traffic_max);
end
guidata(hObject, handles);
