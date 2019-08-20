function [s_hat, idx] = Decoder_1bitML(SP, TR, yq)
Nr = SP.Nr;
M = SP.M;
Nu = SP.Nu;

N = TR.N;
Num1 = TR.Num1;
codebook = TR.codebook;

p1 = Num1./N;
pML = zeros(M^Nu,1);
for sym = 1:M^Nu
    
    % ML-Like Decoding
    p1_sym = p1(:,sym);
    p0_sym = 1-p1_sym;
    
    one = yq == 1;
    temp = zeros(2*Nr,1);
    temp(one) = p1_sym(one);
    temp(~one) = p0_sym(~one);
    
    pML(sym) = prod(temp); % Likelihood probabilities for all candidate symbols
    
end
[~, idx] = max(pML);
code = codebook(:,idx);
s_hat = qammod(code,M,'UnitAveragePower',true);

