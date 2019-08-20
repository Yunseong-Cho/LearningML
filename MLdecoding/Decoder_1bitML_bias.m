function [s_hat, idx] = Decoder_1bitML_bias(SP, TR, yq)
Nu = SP.Nu;
Nr = SP.Nr;
M = SP.M;
bias = SP.bias;

codebook = TR.codebook;
N = TR.N_bias;
Num1 = TR.Num1_bias;

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
    temp(temp == 0) = bias; % set bias probability
    
    pML(sym) = prod(temp); % Likelihood probabilities for all candidate symbols
end
[~, idx] = max(pML);
code = codebook(:,idx);
s_hat = qammod(code,M,'UnitAveragePower',true);

