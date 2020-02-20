clear all; close all; clc;

%Load Data
load('Lumerical_Data_1D.mat')

%Interpolate time and space
t_interp = linspace(t(1),t(end),T);
y_interp = linspace(y(1),y(end),N);

%Make 2D space-time grid
[tt,yy] = meshgrid(t,y);
[tt_interp,yy_interp] = meshgrid(t_interp,y_interp);

%Graph original data space-time data
figure
mesh(yy*1.0e6,tt*1.0e15,abs(e_space_time))
xlabel('ums')
ylabel('fs')
title('Lumerical')

%Interpolate space-time
e_space_time_interp = interp2(tt,yy,e_space_time,tt_interp,yy_interp);

%Graph interoplated space-time
figure
mesh(yy_interp*1.0e6,tt_interp*1.0e15,abs(e_space_time_interp))
xlabel('ums')
ylabel('fs')
title('Interpelated')

% mesh(yy_interp*1.0e6,tt_interp*1.0e15,abs(e_space_time_interp))
mesh(flip(abs(e_space_time_interp),1))
xlabel('time')
ylabel('space')
title('interp')

%Interpolate n
n_interp = interp1(y,n,y_interp);

%Graph interoplated n
figure
plot(y_interp*1.0e6,n_interp)
xlabel('ums')
ylabel('n')
title('interp n')

%Create W matrix for STSN
W = zeros(T,N);
for i = 1:T
    W(i,:) = n_interp.^-2;
end

%Graph weights for STSN
figure
imagesc(W)
xlabel('position')
ylabel('time')
title('weigths for STSN')

%Create input/output field for STSN
t0 = input('Initial time step: ');
tf = input('Final time step: ');
e_input = e_space_time_interp(:,t0);
e_output = e_space_time_interp(:,tf);

%Limit weight distrbution to t0,tf
W = W(t0:tf,:);

%Limit e_space_time
e = e_space_time_interp(:,t0:tf);

%Graph input/output field for STSN
figure
plot(abs(e_input))
hold on
plot(abs(e_output))
hold off
legend('input','output')

%Write CSV file for input and output e
csvwrite('Lumerical_Input_1D.csv',real(e_input))
csvwrite('Lumerical_Output_1D.csv',real(e_output))
csvwrite('Lumerical_Space_Time_1D.csv',real(e))
csvwrite('Lumerical_Weights_1D.csv',W)
