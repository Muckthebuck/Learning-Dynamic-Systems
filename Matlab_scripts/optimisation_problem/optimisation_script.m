load("a_in_set.txt");
a_in_set = -a_in_set; % ARX -> SS just requires a -> -a
load("b_in_set.txt");
step_size_a = 0.01;
step_size_b = 0.01;

min_a = min(a_in_set);
max_a = max(a_in_set);
min_b = min(b_in_set);
max_b = max(b_in_set);

q = 1; r = 1;

%% Brute force: LQR approach

Theta = [a_in_set b_in_set];
n =  length(Theta);
output = zeros(n,2);
Ts = 1/100;

for i=1:n
    a = Theta(i,1); b = Theta(i,2);

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
    else
        disp("unstable!")
    end

end

figure(1); clf; hold on

% ~~~ plot results

% special plants
[~, idx] = max(output(:,2));
scatter3(a_in_set(idx), b_in_set(idx), output(idx,2), 36, 'black', 'Marker','*', 'DisplayName', 'worst-case plant')
[~, idx] = min(output(:,2));
scatter3(a_in_set(idx), b_in_set(idx), output(idx,2), 36, 'black', 'Marker','+', 'DisplayName', 'best-case plant')
idx = find(a_in_set == 0.7 & b_in_set == 1);
scatter3(a_in_set(idx), b_in_set(idx), output(idx,2), 36, 'red', 'Marker','X', 'DisplayName', 'true plant')

% all confidence region plants
scatter3(a_in_set, b_in_set, output(:,2), 36, output(:,1), 'DisplayName', 'objective function scores')
colormap("parula")
cb = colorbar(); 
ylabel(cb,'k','FontSize',16,'Rotation',270)
view(0,90)
xlabel("a")
ylabel("b")
zlabel("J")
legend
title("NB: z value gives J, color value gives k")

% [~, idx] = min(output(:,1));
% fprintf("Lowest k is %.4f at (a,b) = (%.2f,%.2f), J=%.4f\n", ...
%     output(idx,1), a_in_set(idx), b_in_set(idx), output(idx,2))
% 
% [~, idx] = max(output(:,2));
% fprintf("Highest k is %.4f at (a,b) = (%.2f,%.2f), J=%.4f\n", ...
%     output(idx,1), a_in_set(idx), b_in_set(idx), output(idx,2))
% 
% fprintf("\n")
% 
% [~, idx] = min(output(:,2));
% fprintf("Lowest J is %.4f at (a,b) = (%.2f,%.2f), k=%.4f\n", ...
%     output(idx,2), a_in_set(idx), b_in_set(idx), output(idx,1))
% 
% [~, idx] = max(output(:,2));
% fprintf("Highest J is %.4f at (a,b) = (%.2f,%.2f), k=%.4f\n", ...
%     output(idx,2), a_in_set(idx), b_in_set(idx), output(idx,1))
% fprintf("\n")

[~, idx] = min(output(:,1));
min_k = output(idx,1);
[~, idx] = max(output(:,1));
max_k = output(idx,1);
fprintf("Optimal k ranges from %.4f to %.4f\n", min_k, max_k)
[~, idx] = max(output(:,2));
optimal_k = output(idx,1);
associated_a = a_in_set(idx);
associated_b = b_in_set(idx);
fprintf("Optimal k w.r.t minimising worst-case J is %.4f, associated with (a,b) = (%.2f,%.2f)\n", optimal_k, associated_a, associated_b)


%% Brute force: explicit approach

k_vals = linspace(0,0.5,51)';
nk = length(k_vals);
data = zeros(nk,5);

% for each candidate k
for ik=1:nk

    % find the (a,b) from the confidence set which maximises J
    k = k_vals(ik);
    worst_J = 0;
    best_J = inf;
    associated_a = 0;
    associated_b = 0;
    for i=1:n
        a = Theta(i,1); b = Theta(i,2);
        J = (q + r*k^2) / (1 - (a-b*k)^2);
        % fprintf("k=%.4f, a=%.2f, b=%.2f gives J=%.4f\n", k, a, b, J)
        if J < 0
            J = inf;
        end
        if J > worst_J
            worst_J = J;
            associated_a = a;
            associated_b = b;
        end
        if J < best_J
            best_J = J;
        end
    end

    % store results
    data(ik,1) = k;
    data(ik,2) = worst_J;
    data(ik,3) = best_J;
    data(ik,4) = associated_a;
    data(ik,5) = associated_b;
    % NB: associated with the worst/maximum J, not the best/minimum J

end

% [~,idx] = min(data(:,2));
% fprintf("With k=%.4f the upper bound for J is %.4f, lower bound %.4f\n", ...
%     data(idx,1), data(idx,2), data(idx,3))
% fprintf("The upper bound occurs at (a,b) = (%.2f,%.2f)\n", ...
%     data(idx,4), data(idx,5))

