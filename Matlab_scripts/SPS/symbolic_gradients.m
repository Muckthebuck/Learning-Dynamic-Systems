n_states = 2;
n_inputs = 1;
n_output = 2;
n_noise_deg = -1;
C = [0 1; 1 0];
%C = [0 0 0 1; 0 1 0 0];

% Generate symbolic transfer function matrix
[G_zinv, a, b] = observability_tf_zinv(n_states, n_inputs, C);
[H_zinv, H_num, a, c] = observability_H_zinv_poly(n_states, n_output, a, n_noise_deg);
a = fliplr(a);
Y = sym('Y', [n_output, 1]);
U = sym('U', [n_inputs ,1]);
e = H_zinv^-1 *(Y - G_zinv*U);

%generate_diff(G_zinv, a,b,c,'G')
generate_diff(e,a,b,c,'\epsilon')

%% closed loop
F =  sym('F', [n_inputs, n_output]);
L =  sym('L', [n_inputs, n_output]);
R =  sym('R', [n_output, 1]);
e_c_o = H_zinv^-1 *(Y - G_zinv*(F*Y -L*R));
generate_diff(e_c_o,a,b,c,'\epsilon_o')

%%

function generate_diff(e,a,b,c,name)
    syms z u
    all_syms = [a, b(:).', c(:).'];  % row vector of all a_i,  b_i, c_i symbols
    disp(all_syms)
    [nrows, ncols] = size(e);

    for i = 1:nrows
        for j = 1:ncols
            %fprintf('\n--- Derivatives of G(%d,%d)(z⁻¹) ---\n', i, j);
            for k = 1:length(all_syms)
                var = all_syms(k);
                de = diff(e(i,j), var);
                % Display LaTeX-formatted result using z⁻¹ instead of u
                de_latex = latex(simplify(de));
                fprintf('&\\frac{\\partial %s(%d,%d)}{\\partial %s} = %s\\\\\n',name, i, j, char(var), de_latex);
            end 
        end
    end
end

% ==== Function ====
function [G_zinv, a, b] = observability_tf_zinv(n_states, n_inputs, C)
    % Create symbolic variables
    a = sym('a', [1 n_states]);            % Creates a1, ..., an
    a = fliplr(a);                         % Reorder to a_n, ..., a_1

    b = sym('b', [n_states*n_inputs, 1]);   % b(i,j) = b_ij
    b = reshape(b, [n_states,n_inputs]);
    z = sym('z');                          % complex variable z
    u = sym('u');                          % u = z⁻¹

    % Companion matrix (observability form)
    A_obs = sym(zeros(n_states));
    A_obs(2:end, 1:end-1) = eye(n_states - 1);
    A_obs(:, end) = -transpose(a);         % Uses a_n to a_1 in last column

    % Flip B_obs for observability canonical form
    B_obs = flipud(b);

    % Ensure C is symbolic
    C = sym(C);

    % G(z)
    G_z = C * ((z * eye(n_states) - A_obs)^(-1)) * B_obs;

    % Substitute z = 1/u to get G(z⁻¹)
    G_zinv = simplify(subs(G_z, z, 1/u));
end
function [H_zinv, H_num, a, c] = observability_H_zinv_poly(n_states, q, a, n_v)
    % Create symbolic variables
    z = sym('z');
    u = sym('u');  % z⁻¹

    % Define A(z⁻¹) = 1 + a1 u + a2 u^2 + ...
    A_zinv = 1;
    for k = 1:n_states
        A_zinv = A_zinv + a(n_states - k + 1) * u^k;
    end

    % Create 1D symbolic vector for all c_ij coefficients
    c = sym('c', [1, q * (n_v + 1)]);  % row vector

    % Build diagonal H_num
    H_num = sym(zeros(q));
    for i = 1:q
        if n_v >=0
            Ci_zinv = 0;
            for j = 0:n_v
                idx = (i - 1) * (n_v + 1) + j + 1;  % Compute flat index
                Ci_zinv = Ci_zinv + c(idx) * u^j;
            end
            H_num(i, i) = simplify(Ci_zinv);
        else
            H_num(i,i) = 1;
        end

    end

    % Final H(z⁻¹)
    H_zinv = simplify(H_num / A_zinv);
end

