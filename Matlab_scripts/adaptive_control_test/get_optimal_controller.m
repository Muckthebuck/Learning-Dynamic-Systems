function [k,J,a,b] = get_optimal_controller(a_in_set, b_in_set, q, r)
    
    % a_in_set: parameter "a" in the SS model
    % b_in_set: parameter "b" in the SS model

    assert(length(a_in_set) == length(b_in_set))
    n =  length(a_in_set);
    output = zeros(n,4);

    for i=1:n
        a = a_in_set(i);
        b = b_in_set(i);
    
        % solve the DARE and compute optimal gain k
        % k = lqr(ss(a,b,1,0,Ts), q, r);
        % k = dlqr(a,b,q,r,0);
        [k, ~] = dlqr_custom(a,b,q,r);
    
        % compute cost J
        J = (q + r*k^2) / (1 - (a-b*k)^2);
        
        % store results
        if (abs(a-b*k) < 1)
            output(i,1) = k;
            output(i,2) = J;
            output(i,3) = a;
            output(i,4) = b;
        else
            disp("unstable!")
        end
    
    end

    [~,idx] = max(output(:,2));
    k = output(idx,1);
    J = output(idx,2);
    a = output(idx,3);
    b = output(idx,4);
    % disp(output)
    
end