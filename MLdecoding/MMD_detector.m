function [s_hat] = MMD_detector(SP, TR, yq)

Nu = SP.Nu;
M = SP.M;

% yq = sign(real(y)) + 1j*sign(imag(y));
temp_dist = zeros(1,M^Nu); 

for sym = 1:M^Nu  
    for n = 1:TR.N_vec(sym)
        temp_dist(sym) = temp_dist(sym) + norm(yq-TR.possible_vector(sym,:,n)',2)*TR.pmf(sym, n);
    end
end
[~, idx] = min(temp_dist);
code = TR.codebook(:,idx);

s_hat = qammod(code,M,'UnitAveragePower',true);
end

