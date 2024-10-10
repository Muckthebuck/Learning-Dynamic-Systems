%% description



% adaptive control scheme

% 1) generate SPS confidence region using open loop plant
% 2) calculate optimal controller given confidence region
% 3) run the closed loop plant using this optimal controller
% 4) use this new input/output data to rerun SPS and generate new
% confidence region
% 5) repeat steps 2 to 4


clear
close all

%% parameters


% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ plant parameters
theta_real = [0.4, 0.3, 0]
A = [1, -theta_real(1)];   % AR coefficients
B = [0 theta_real(2)];       % MA coefficients 
% B = [theta_real(2) 0];       % MA coefficients 
C = [1, theta_real(3)];       % input coefficients
D = 0;           % Constant term
noiseVar = 0.0;  % Variance of the noise


% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SPS parameters
m = 20;   % Integer m
q = 1;     % Integer q 
p = 1-q/m    % Confidence probability p = 1 − q/m
T = 1;

% Define the initial range of a and b, grid resolution
grid_res = 0.1;
a_values = grid_res:grid_res:1;
b_values = grid_res:grid_res:1;
% a_values = -1:grid_res:1;
% b_values = -1:grid_res:1;
% a_values = 0.42:grid_res:0.48;
% b_values = 0.29:grid_res:0.37;
% a_values = 0.2:grid_res:0.6;
% b_values = 0.1:grid_res:0.5;


% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ initialize the parallel pool (optional)
if isempty(gcp('nocreate'))
    parpool;
end

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ optimal controller parameters

Q = 1;
R = 1;


%% iteration 1

n_iters = 3;

figure(1); clf; hold on
subplot(3,1,1); hold on; legend
subplot(3,1,2); hold on
subplot(3,1,3); hold on

figure(2); clf; hold on
title('Confidence Region')
xlabel('a')
ylabel('b')
xlim([0 1])
ylim([0 1])
grid on
legend

for i=1:n_iters
    
    % Update plant + controller
    if i==1
        disp("first iteration, open loop plant")
        K = 0;
    else
        % Update the SPS confidence region considered

        % Strategy 0 - leave the considered ab region as is
        % (no code needed)
    
        % Strategy 1 - rectangle enclosing confidence region
        % [a_values, b_values] = update_ab_considered(conf_region, grid_res);

    end

    % Update the controller with new K
    armaxGen = ARMAXGenerator(A, B, C, D, @step_controller_K, K);

    % Update seed and n (number of samples)
    rng(i)
    n = update_n(i);
    sps = SPS(p, m, q, n, T);

    % Generate and plot data
    e = randn(n, 1)*sqrt(noiseVar); % Noise sequence
    [y, u, e] = armaxGen.generateData(n,e);
    armaxGen.plotData(y, u, e, 1);
    
    % Get the confidence region
    conf_region = get_confidence_region(sps, a_values, b_values, y, u, e, K);
    if(isempty(conf_region))
        disp("No points in confidence region!")
        return
    end
    a_in_set = conf_region(:,1);
    b_in_set = conf_region(:,2);
    
    % Plot the confidence region
    if ~isempty(conf_region)
        figure(2)
        z_vals = 0.1*ones(length(conf_region),1)*i; % height = iteration number
        scatter3(a_in_set, b_in_set, z_vals, 'DisplayName', sprintf('SPS confidence region iter %d', i))
    else
        disp('No points in the confidence region.')
    end
    
    % Get the optimal controller
    % ARMAX (a,b) -> SS (a,b) is assumed a direct mapping for now
    [K,J,a,b] = get_optimal_controller(a_in_set, b_in_set, Q, R);
    K

end

figure(2)
scatter3(theta_real(1), theta_real(2), 0.1*n_iters, 'g+', 'DisplayName', 'True theta')  % Add the true theta









%% helper functions


function [a_values, b_values] = update_ab_considered(conf_region, grid_res)

    % Strategy 1
    %   - new ab is a rectangle enclosing the previous confidence region
    %   - keep the same grid resolution
    min_a = min(conf_region(:,1));
    max_a = max(conf_region(:,1));
    min_b = min(conf_region(:,2));
    max_b = max(conf_region(:,2));
    a_values = min_a:grid_res:max_a;
    b_values = min_b:grid_res:max_b;

end


function n = update_n(i)
    % n = i*200; % increase #samples each iteration
    n = 400; % keep #samples constant for each iteration
end


function conf_region = get_confidence_region(sps, a_values, b_values, y, u, e, K)

    % Generate random signs and permutation
    sps = sps.generateRandomSignsAndPermutation();
    
    % Get the number of iterations for as
    num_a = numel(a_values);
    
    % Preallocate a cell array to store results from each worker
    conf_region_cell = cell(num_a, 1);
    
    parfor i = 1:num_a
        a = a_values(i); % Map index i to the corresponding value of a
        local_conf_region = [];
        
        for b = b_values
            c = 0; % Fixed value for c
            A_hat = [1, -a];   % AR coefficients
            B_hat = [0, b];    % MA coefficients
            % B_hat = [b, 0];    % MA coefficients
            C_hat = [1, c];    % input coefficients
            D_hat = 0;         % Constant term
            
            % Check the condition using the sps_indicator function
            if abs(a - 0.4) < 1e-5 && abs(b - 0.3) < 1e-5
                if sps.sps_indicator(y, u, A_hat, B_hat, C_hat, D_hat, @step_controller_K, K, @createFeatureMatrix) == 1
                    disp("true theta included")
                else
                    disp("true theta not included")
                end
            end
            if sps.sps_indicator(y, u, A_hat, B_hat, C_hat, D_hat, @step_controller_K, K, @createFeatureMatrix) == 1
                local_conf_region = [local_conf_region; a, b, c];
            end
        end
        
        % Store the result in the cell array
        conf_region_cell{i} = local_conf_region;
    end
    
    % Concatenate all results from the cell array into the final conf_region
    conf_region = vertcat(conf_region_cell{:});

end


function u_out = step_controller_K(y,u,t,K)
    
    amp = 0.1;

    % swept sine?
    % chirp?
    % % excitation_signal = amp * (t>10); % step
    % excitation_signal = amp * (mod(t,10) == 0); % impulses
    % excitation_signal = amp * sign(sin(2*pi*t/100)); % square wave
    excitation_signal = randn(1) * 1;
    
    u_out = -K*y(t-1) + excitation_signal;
    % u_out = excitation_signal;

    % u_out = 0.1 * randn(1); % noise

end


function phi = get_phi(y, u, nt, t)
    phi = [y(t-1), u(t-1)];
end


function X = createFeatureMatrix(y, u, nt)
    % Ensure y and u have the same length
    if length(y) ~= length(u)
        error('Vectors y and u must have the same length.');
    end
    
    % Number of data points
    N = length(y);
    
    % Initialize the feature matrix X
    % X will have (N) rows and 4 columns corresponding to  [-y(t-1), u(t-1), nt(t), nt(t-1)]
    X = zeros(N, 2);
    
    % Fill the feature matrix
    for t = 2:N
        X(t-1, :) = get_phi(y,u, nt, t);
    end
end


