function [s_hat] = MCD_detector(SP, TR, yq)

Nu = SP.Nu;
M = SP.M;

% yq = sign(real(y)) + 1j*sign(imag(y));
temp_dist = zeros(1,M^Nu); 

for sym = 1:M^Nu    
    temp_dist(1,sym) = norm(yq-TR.centroid(:,sym), 2);
end
[~, idx] = min(temp_dist);
code = TR.codebook(:,idx);

s_hat = qammod(code,M,'UnitAveragePower',true);
end

