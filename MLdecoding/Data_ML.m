function [ML] = Data_ML(i, SP, H,  TR)

% SP.Nr = SP.Nr(i);
% SP.Nu = SP.Nu(i);
% SP.M = SP.M(i);
SP.p = SP.p(i);
SP.p_dither = SP.p_dither(i);
% SP.p_ici = SP.p_ici(i);

D = SP.D;
Nd = SP.Nd;
Nr = SP.Nr;
Ntr = SP.Ntr;
M = SP.M;
Nu = SP.Nu;
p = SP.p;
p_dither = SP.p_dither;
CRC1 = SP.CRC1;

%% INITIALIZATION (Likelihood Functions)

codebook = TR.codebook;
x_initial = sqrt(p)*qammod(codebook, M, 'UnitAveragePower', true); % Generate symbol

p0 = zeros(2*Nr,M^Nu);
Ncdf = @(t) 1/sqrt(2*pi)*exp(-t.^2/2);
Hr = [real(H), -imag(H); imag(H), real(H)];

% Compute Opt 1-bit likelihood functions
for sym = 1:M^Nu
    xr = [real(x_initial(:,sym)); imag(x_initial(:,sym))];
    mu = Hr*xr;
    for n = 1:2*Nr
        p0(n, sym) = integral(Ncdf, -inf, -sqrt(2)*mu(n)); % p[y_i = -1]
    end
end
TR.p0 = p0;

% Compute Dithering 1-bit likelihood functions
est_snr_dB = TR.est_snr_dB; % Estimated SNR in dB
est_snr = 10^(est_snr_dB/10);
N0 = 1;%p/est_snr; % Estimated noise variance (supposed to be 1)

error_count_opt = 0;
error_count_ZF = 0;
error_count = 0;
error_count_bias = 0;
error_count_dither = 0;
error_count_emld = 0;
error_count_mmd = 0;
error_count_mcd = 0;
switch SP.CDF
    case 'exact' % exact CDF
        Num1d = TR.Num1_dither;
        p1d = Num1d./Ntr; % p[y_i = 1 | dithered ]
        for sym = 1:M^Nu
            phi = - sqrt(N0/2 + p_dither/2)* qfuncinv(p1d(:,sym)); % phi = \sqrt(p)fs
            p1d(:,sym) = qfunc(-phi/sqrt(N0/2));
        end
        TR.p1_dither = p1d; % p[y_i = 1]
        
    case 'approx' % approximated CDF
        Num1d = TR.Num1_dither;
        p1d = Num1d./Ntr;
        for sym = 1:M^Nu
            idx = p1d(:,sym) < 0.5;
            hx1 = -sqrt(N0/2+ p_dither/2)*10/log(41)*log(1-log((-log(1-p1d(idx,sym)))/log(2))/log(22));
            hx0 = -sqrt(N0/2+ p_dither/2)*10/log(41)*log(1-log((-log(p1d(~idx,sym)))/log(2))/log(22));
            p1d(idx,sym) = 1-2.^(-22.^(1-41.^(-2/N0*hx1/10)));
            p1d(~idx,sym) = 2.^(-22.^(1-41.^(-2/N0*hx0/10)));
        end
        TR.p1_dither = p1d;
end

%% TRANSMISSION & DETECTION
cnt_crc = 0;
cnt_crc_bias = 0;
cnt_crc_dither = 0;
for d = 1:D
    cb_idx = zeros(Nd,1);
    cb_idx_bias = zeros(Nd,1);
    cb_idx_dither = zeros(Nd,1);
    Yq = zeros(2*Nr, Nd);
    
    [x_crc, s_crc] = Transmitter_ML(SP); % Tx signal x = sqrt(p)s
    for t = 1:Nd
        
        x = x_crc(:,t); s = s_crc(:,t);
        [DC] = Receiver_ML(SP, H, TR, x); % Rx and decode
        Yq(:,t) = DC.yq; % Quantized data signal (saved for post update)
        
        s_hat_opt = DC.s_hat_opt;
        error_opt = sum(s_hat_opt ~= s);
        error_count_opt = error_count_opt + error_opt;
        
        s_hat_ZF = DC.s_hat_ZF;
        error_ZF = sum(s_hat_ZF ~= s);
        error_count_ZF = error_count_ZF + error_ZF; 
        
        s_hat = DC.s_hat;
        cb_idx(t) = DC.cb_idx;
        error = sum(s_hat ~= s);
        error_count = error_count + error;
        
        s_hat_bias = DC.s_hat_bias;
        cb_idx_bias(t) = DC.cb_idx_bias;
        error_bias = sum(s_hat_bias ~= s);
        error_count_bias = error_count_bias + error_bias;
        
        s_hat_dither = DC.s_hat_dither;
        cb_idx_dither(t) = DC.cb_idx_dither;
        error_dither = sum(s_hat_dither ~= s);
        error_count_dither = error_count_dither + error_dither;
        
        s_hat_emld = DC.s_hat_emld;
        error_emld = sum(s_hat_emld ~= s);
        error_count_emld = error_count_emld + error_emld;
        
        s_hat_mmd = DC.s_hat_mmd;
        error_mmd = sum(s_hat_mmd ~= s);
        error_count_mmd = error_count_mmd + error_mmd;
        
        s_hat_mcd = DC.s_hat_mcd;
        error_mcd = sum(s_hat_mcd ~= s);
        error_count_mcd = error_count_mcd + error_mcd;
    end
    
    % CRC check
    crc = 0;
    crc_bias = 0;
    crc_dither = 0;
    for u = 1:Nu
        temp1 = de2bi(codebook(u,cb_idx),log2(M))';
        temp2 = de2bi(codebook(u,cb_idx_bias),log2(M))';
        temp3 = de2bi(codebook(u,cb_idx_dither),log2(M))';
        
        s_crc = temp1(:);
        s_crc_bias = temp2(:);
        s_crc_dither = temp3(:);
        [~, c1ML] = lteCRCDecode(s_crc, CRC1);
        [~, c1ML_bias] = lteCRCDecode(s_crc_bias, CRC1);
        [~, c1ML_dither] = lteCRCDecode(s_crc_dither, CRC1);
        
        crc = crc + c1ML;
        crc_bias = crc_bias + c1ML_bias;
        crc_dither = crc_dither + c1ML_dither;
    end
    
    % Post Update
    if crc == 0
        cnt_crc = cnt_crc + 1;
        TR = ML_update(SP, TR, Yq, cb_idx, 0);
    end
    
    if crc_bias == 0
        cnt_crc_bias = cnt_crc_bias + 1;
        TR = ML_update(SP, TR, Yq, cb_idx_bias, 1);
    end
    
    if crc_dither == 0
        cnt_crc_dither = cnt_crc_dither + 1;
        TR = ML_update(SP, TR, Yq, cb_idx_dither, 2);
    end
end

ML.error_opt = error_count_opt;
ML.error_zf = error_count_ZF;
ML.error = error_count;
ML.error_bias = error_count_bias;
ML.error_dither = error_count_dither;
ML.error_emld = error_count_emld;
ML.error_mmd = error_count_mmd;
ML.error_mcd = error_count_mcd;

ML.cnt_crc_bias = cnt_crc_bias;
ML.cnt_crc_dither = cnt_crc_dither;
% TR.d = cnt_m1ML_dither;
