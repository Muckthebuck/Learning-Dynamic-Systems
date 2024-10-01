classdef SPS
    properties
        p      % Confidence probability
        m      % Integer m such that p = 1 − q/m
        q      % Integer q such that p = 1 − q/m
        n      % Number of data points
        T      % block SPS window size
        alpha  % Random signs
        pi     % Random permutation
    end
    
    methods
        function obj = SPS(p, m, q, n, T)
            % Constructor to initialize the SPS parameters
            obj.p = p;
            obj.m = m;
            obj.q = q;
            obj.n = n;
            obj.T = T;
        end
        
        function obj = generateRandomSignsAndPermutation(obj)
            % Generate and return random signs and permutation
             % Generate random signs
            %obj.alpha = randi([0, 1], obj.m-1, obj.n) * 2 - 1; % Random signs {-1, 1}
            
            % block SPS
            obj.alpha = obj.generate_alpha_matrix(obj.m, obj.n, obj.T);
            % Generate a random permutation
            obj.pi = randperm(obj.m);
        end
        
        function alpha = generate_alpha_matrix(~, m, n, T)
            % Generate an m by n matrix of modified random signs for SPS method
            % m: Number of random sequences (rows)
            % n: Length of each sequence (columns)
            % T: Desired block size, where the same sign is held constant

            % Adjust T to be the closest divisor of n if needed
            if mod(n, T) ~= 0
                divisors = 1:n;
                valid_divisors = divisors(mod(n, divisors) == 0);
                [~, idx] = min(abs(valid_divisors - T));
                T = valid_divisors(idx);
                fprintf('Adjusted T to %d, closest divisor of n.\n', T);
            end

            % Number of blocks in each sequence
            num_blocks = n / T;

            % Generate random signs for each block for each sequence
            block_signs = 2 * randi([0, 1], m, num_blocks) - 1;

            % Repeat each sign within a block across the entire sequence
            alpha = repelem(block_signs, 1, T);
        end
        
        function theta_hat = leastSquaresEstimator(~, X, y)
            % Least Squares Estimator
            % X - Design matrix
            % y - Output vector
            
            % Ensure X is full rank for least squares estimation
            if rank(X) < size(X, 2)
                error('Design matrix X is rank deficient.');
            end
            
            % Compute least squares estimate
            theta_hat = (X' * X) \ (X' * y);
        end
        
        function [theta_hat] = estimateParameters(obj, X, y)
            % Estimate parameters using least squares
            % X - Design matrix
            % y - Output vector
            
            % Check if design matrix X and output vector y are valid
            if size(X, 1) ~= length(y)
                error('Design matrix X and output vector y must have the same number of rows.');
            end
            
            % Compute parameter estimates
            theta_hat = obj.leastSquaresEstimator(X, y);
        end
        
        function y_pred = predictor(~, X, theta_hat)
            % Predictor to compute predicted values using estimated parameters
            % X - Design matrix for prediction
            % theta_hat - Estimated parameters
            
            % Ensure that the design matrix X has the same number of columns as theta_hat
            if size(X, 2) ~= length(theta_hat)
                error('Design matrix X and estimated parameters theta_hat have incompatible dimensions.');
            end
            
            % Compute the predicted values
            y_pred = X * theta_hat;
        end
        
        function is_in_conf_region = sps_indicator(obj, y,u, A_star, B_star, C_star, D_star, controller, createFeatureMatrix)
            
            % Compute A(z^-1)y_t
            A_yt = filter(A_star, 1, y);
            % Compute B(z^-1)u_t
            B_ut = filter(B_star, 1, u);
            % Compute the difference A(z^-1)y_t - B(z^-1)u_t
            diff_AB = A_yt - B_ut;
            % reconstructed noise C^-1(A(z^-1)y_t - B(z^-1)u_t)
            nt_theta = filter(1, C_star, diff_AB);
            % build m-1 sequences of sign perturbed prediction errors
            sp_errors = obj.alpha.*nt_theta';
          
            
            % Create an instance of ARMAXGenerator
            armaxGen_bar = ARMAXGenerator(A_star, B_star, C_star, D_star, controller);
            
            S = zeros(1,obj.m);
            %[y_bar, u_bar, ~] = armaxGen_bar.generateData(obj.n,nt_theta');
            % Make predictions with new data

            X = createFeatureMatrix(y, u, nt_theta);
            R = X' * X / obj.n;
            S(1) = norm(R^(-1/2) * X'*nt_theta/obj.n);

            for k = 2:obj.m 
                [y_bar, u_bar, ~] = armaxGen_bar.generateData(obj.n,sp_errors(k-1,:)');
                X_bar =  createFeatureMatrix(y_bar,u_bar, sp_errors(k-1,:));
                R =  X_bar'* X_bar /obj.n;
                S(k) = norm(R^(-1/2) * X_bar'*sp_errors(k-1,:)'/obj.n);
                %armaxGen.plotData(y_bar(k,:), u_bar(k,:), sp_errors(k,:)');
            end

            % Combine S and Pi into a matrix

            SP = [S(:), obj.pi(:)];

            % Sort SM based on S with Pi as a tiebreaker and keep track of original indices
            [~, sort_order] = sortrows(SP, [1, 2]);

            % Find the new index of the first element of the original S
            new_index = find(sort_order == 1);

            is_in_conf_region = new_index<=obj.m-obj.q;
        end
        
    end
end
