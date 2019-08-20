function [s_hat] = Decoder_1ML_opt(SP, TR, yq)
Nu = SP.Nu;
Nr = SP.Nr;
M = SP.M;
p0 = TR.p0;
pML = zeros(M^Nu,1);
codebook = TR.codebook;

for sym = 1:M^Nu
   
    % ML Decoding
    p0_sym = p0(:,sym);
    p1_sym = 1-p0_sym;
    
    one = yq == 1;
    temp = zeros(2*Nr,1);
    temp(one) = p1_sym(one);
    temp(~one) = p0_sym(~one);
    
    pML(sym) = prod(temp); % Likelihood probabilities for all candidate symbols
end
[~, idx] = max(pML);
s_hat = qammod(codebook(:,idx), M, 'UnitAveragePower', true);

