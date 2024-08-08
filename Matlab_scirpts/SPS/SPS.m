classdef SPS
    properties
        p      % Confidence probability
        m      % Integer m such that p = 1 − q/m
        q      % Integer q such that p = 1 − q/m
        n      % Number of data points
        alpha  % Random signs
        pi     % Random permutation
    end
    
    methods
        function obj = SPS(p, m, q, n)
            % Constructor to initialize the SPS parameters
            obj.p = p;
            obj.m = m;
            obj.q = q;
            obj.n = n;
        end
        
        function obj = generateRandomSignsAndPermutation(obj)
            % Generate and return random signs and permutation
             % Generate random signs
            obj.alpha = randi([0, 1], obj.m-1, obj.n) * 2 - 1; % Random signs {-1, 1}
            % Generate a random permutation
            obj.pi = randperm(obj.m);
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
        
        
    end
end
