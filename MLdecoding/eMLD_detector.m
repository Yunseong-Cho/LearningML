function [s_hat] = eMLD_detector(SP, TR, yq)

Nu = SP.Nu;
M = SP.M;

% yq = sign(real(y)) + 1j*sign(imag(y));

dist = [];
min_dist = inf;
sum_dist = zeros(1,M^Nu);

%% Obtaining R_m and storing each distance for each possible vector
for sym = 1:M^Nu
    for n = 1:TR.N_vec(sym)
        dist(sym, n) = norm(yq-TR.possible_vector(sym,:,n)',2);
    end
    min_dist = min([min_dist, dist(sym, 1:TR.N_vec(sym))]);
end

%% Compare the PMF of possible vectors that correspoing distance is same as R_m
for sym = 1:M^Nu
    for n = 1:TR.N_vec(sym)
        if dist(sym,n) == min_dist
            sum_dist(sym) = sum_dist(sym) + TR.pmf(sym,n);
        end
    end
end
[~, idx] = max(sum_dist);
code = TR.codebook(:,idx);

s_hat = qammod(code,M,'UnitAveragePower',true);
end

