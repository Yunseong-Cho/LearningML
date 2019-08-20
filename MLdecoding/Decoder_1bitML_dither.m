function [s_hat, idx] = Decoder_1bitML_dither(SP, TR, yq)
Nu = SP.Nu;
Nr = SP.Nr;
M = SP.M;
bias = SP.bias;
a = SP.a_update;
codebook = TR.codebook;
p1d = TR.p1_dither;

Num1 = TR.Num1_dither_post; % For post update
N = TR.N_dither + 0.000001; % For post update
p1 = Num1./N;               % estimated likelihood function from decoded data

pML = zeros(M^Nu,1);
for sym = 1:M^Nu
    
    p1_sym = p1d(:,sym);
    
    % POST UPDATE
    n = round(N(sym));
    if n ~= 0
        p1_sym = a(n)*p1_sym + (1-a(n))*p1(:,sym);
    end
    p0_sym = 1-p1_sym;
    
    one = yq == 1;
    temp = zeros(2*Nr,1);
    temp(one) = p1_sym(one);
    temp(~one) = p0_sym(~one);
    temp(temp == 0) = bias;
    
    pML(sym) = prod(temp); % Likelihood probabilities for all candidate symbols
end
[~, idx] = max(pML);
code = codebook(:,idx);
s_hat = qammod(code,M,'UnitAveragePower',true);

