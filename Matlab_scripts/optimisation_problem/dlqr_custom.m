function [K,P] = dlqr_custom(A,B,Q,R)

% ideally, this tolerance is automatically computed based on A,B
% for now leave it manually set
tolerance = 1e-12;
max_iters = 1000;

% initial guess
P = Q;

% iterative solver
for i = 1:max_iters
    P_next = A'*P*A - (A'*P*B) * inv(R + B'*P*B) * (B'*P*A) + Q;

    if norm(P_next - P, 'fro') < tolerance
        break
    end

    P = P_next;
end

if i == max_iters
    disp("Error converging")
end

% calculate and return K along with P
% https://au.mathworks.com/help/control/ref/lti.dlqr.html#mw_6e129893-5cab-4a33-a5b8-d8b145402851
K = inv(R + B'*P*B) * B'*P*A;

end



