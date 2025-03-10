% MIMO Cart-Pendulum System with LQR Control
% Mukul Chodhary - 2025

clear; clc; close all;

% Physical Parameters
m_c = 1.0;     % Cart Mass (kg)
m_p = 0.2;     % Pendulum Mass (kg)
l = 0.5;       % Pendulum Length (m)
g = 9.81;      % Gravity (m/s^2)
b = 0.1;       % Damping Coefficient

% State-Space Representation
A = [0 1 0 0;
     0 -b/m_c (m_p*g)/m_c 0;
     
     0 0 0 1;
     0 -b/(l*m_c) (m_c + m_p)*g/(l*m_c) 0];

B = [0; 1/m_c; 0; 1/(l*m_c)];
C = eye(4);  % Full State Measurement
D = zeros(4, 1);

% LQR Controller Design
Q = diag([10, 1, 10, 1]);  % Penalize position and angle
R = 0.1;                   % Penalize control effort

[K, ~, ~] = lqr(A, B, Q, R);
disp('LQR Gain Matrix:');
disp(K);

% Compute Transfer Function (G(z^-1)) from state-space
sys = ss(A, B, C, D);  % State-Space System
G_z = tf(sys);  % Transfer Function in continuous-time

% For discrete-time system (assuming sample time dt)
sys_d = c2d(sys, dt);  % Convert to discrete-time system
G_z_d = tf(sys_d);  % Transfer Function in discrete-time

disp('Transfer Function G(z^-1):');
disp(G_z_d);

% Simulation Parameters
dt = 0.01;    % Time Step
T = 10;       % Simulation Time
t = 0:dt:T;   % Time Vector

% Initial Conditions
x0 = [0.1; 0; pi/6; 0];  % [Cart Position, Velocity, Pendulum Angle, Angular Velocity]
x = x0; 
X = zeros(length(t), length(x0));

% For Residual Calculation
theta = 0.5;  % Model parameter
z_inv = 1;    % Assume z^-1 = 1 for simplicity in the discrete system
H = 1 - theta * z_inv;  % Example Transfer Function H(z^-1; theta)
H_inv = 1 / H;  % Inverse of H(z^-1)

% Initialize residuals array
epsilon = zeros(length(t), 1);

% Simulation Loop
for i = 1:length(t)
    % Control Law (LQR)
    u = -K * x;           % Control Law
    
    % State Dynamics (from the system model)
    x_dot = A * x + B * u;    % State Dynamics
    x = x + x_dot * dt;   % Euler Integration
    X(i, :) = x';

    % Residual Calculation
    % Predicted output from the system
    predicted_output = C * x; % Assuming full-state observation for simplicity
    residual = predicted_output - C * x;  % This would typically be Y_t - G(z^-1; theta) * U_t
    epsilon(i) = H_inv * residual;  % Apply the inverse transfer function to residual
end

% Plot Results
figure;
subplot(2, 1, 1);
plot(t, X(:, 1), 'b', 'LineWidth', 1.5);
xlabel('Time [s]');
ylabel('Cart Position [m]');
title('Cart Position');
grid on;

subplot(2, 1, 2);
plot(t, X(:, 3), 'r', 'LineWidth', 1.5);
xlabel('Time [s]');
ylabel('Pendulum Angle [rad]');
title('Pendulum Angle');
grid on;

sgtitle('MIMO Cart-Pendulum System with LQR Control');

% Plot Residuals
figure;
plot(t, epsilon, 'g', 'LineWidth', 1.5);
xlabel('Time [s]');
ylabel('Residual \epsilon_t(\theta)');
title('Residuals for MIMO Cart-Pendulum System');
grid on;

% Animation (Optional)
figure;
for i = 1:10:length(t)
    clf;
    hold on;
    plot([-2, 2], [0, 0], 'k', 'LineWidth', 2);  % Ground
    rectangle('Position', [X(i, 1)-0.1, 0, 0.2, 0.1], 'FaceColor', 'b'); % Cart
    pend_x = X(i, 1) + l * sin(X(i, 3));
    pend_y = l * cos(X(i, 3));
    plot([X(i, 1), pend_x], [0, pend_y], 'r', 'LineWidth', 2);  % Pendulum
    plot(pend_x, pend_y, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
    axis([-2 2 -1 1]);
    xlabel('Position');
    ylabel('Height');
    title('Cart-Pendulum Animation');
    drawnow;
end
