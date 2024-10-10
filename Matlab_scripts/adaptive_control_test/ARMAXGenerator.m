%% A(q)y(t)=B(q)u(t)+C(q)e(t)+D
classdef ARMAXGenerator
    properties
        A  % AR coefficients (polynomial A(q))
        B  % Exogenous input coefficients (polynomial B(q))
        C  % MA coefficients (polynomial C(q))
        D  % Constant term
        input_controller % Controller function for u(t) % inputs y,u,t
        input_controller_args % typically the gain values
        order  % Order of the AR, MA, and Exogenous terms
    end
    
    methods
        function obj = ARMAXGenerator(A, B, C, D, input_controller, input_controller_args)
            % Constructor to initialize the ARMAX model parameters
            obj.A = A;
            obj.B = B;
            obj.C = C;
            obj.D = D;
            obj.order = max([length(A), length(B), length(C)]) - 1;
            obj.input_controller = input_controller;
            obj.input_controller_args = input_controller_args;
        end
        
        function [y, u, e] = generateData(obj, N, e)
            % Generate data using the ARMAX model
            % N - number of data points
            % u -  input signal (optional)
            
            u = zeros(N, 1); % Initialize input array
            
            y = zeros(N, 1); % Output signal
            
            % Generate the output data using the ARMAX model
            for t = obj.order + 1:N
                % AR part
                ar_sum = 0;
                for k = 2:length(obj.A)
                    if t - k + 1 > 0
                        ar_sum = ar_sum + obj.A(k) * y(t - k + 1);
                    end
                end
                
                % Exogenous input part
                exo_sum = 0;
                for k = 1:length(obj.B)
                    if t - k > 0
                        exo_sum = exo_sum + obj.B(k) * u(t - k);
                    end
                end
                
                % MA part
                ma_sum = 0;
                for k = 1:length(obj.C)
                    if t - k + 1 > 0
                        ma_sum = ma_sum + obj.C(k) * e(t - k + 1);
                    end
                end
                
                % Compute the output
                y(t) = (-ar_sum + exo_sum + ma_sum + obj.D) / obj.A(1);
                u(t) = obj.input_controller(y,u,t,obj.input_controller_args);
            end
        end
        
        function plotData(obj, y, u, e, fig_num)
            % Plot the generated data
            % y - output signal
            % u -  input signal
            % e - noise signal
            
            figure(fig_num);
            subplot(3, 1, 1);
            plot(u);
            title('Input Signal');
            xlabel('Time');
            ylabel('u(t)');
            
            subplot(3, 1, 2);
            plot(e);
            title('Noise Signal');
            xlabel('Time');
            ylabel('e(t)');
            
            subplot(3, 1, 3);
            plot(y);
            title('Generated Output Signal');
            xlabel('Time');
            ylabel('y(t)');
        end
    end
end
