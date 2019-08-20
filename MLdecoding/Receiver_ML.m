function [DC] = Receiver_ML(SP, H, TR, x)

Nr = SP.Nr;

% Received signal
n = 1/sqrt(2)*(randn(Nr,1) + 1j*randn(Nr,1));
y = H*x + n; 

% 1-bit Quantization
tempr = sign(real(y));
tempi = sign(imag(y));
yq_real = (tempr == 1);
yq_imag = (tempi == 1);

yq = zeros(2*Nr,1);
yq(1:Nr) = yq_real;
yq(Nr+1:2*Nr) = yq_imag; % Quantized signal vector with 1s 0s

% ML Decoders 
[s_hat_opt] = Decoder_1ML_opt(SP, TR, yq); 
[s_hat_ZF] = ZF_detector(SP, H, y);
[s_hat, cb_idx] = Decoder_1bitML(SP, TR, yq); 
[s_hat_bias, cb_idx_bias] = Decoder_1bitML_bias(SP, TR, yq); 
[s_hat_dither, cb_idx_dither] = Decoder_1bitML_dither(SP, TR, yq);
% Yunseong part
[s_hat_emld] = eMLD_detector(SP,TR,yq);
[s_hat_mmd] = MMD_detector(SP,TR,yq);
[s_hat_mcd] = MCD_detector(SP,TR,yq);

DC.s_hat_opt = s_hat_opt;
DC.s_hat_ZF = s_hat_ZF;
DC.s_hat = s_hat;
DC.s_hat_bias = s_hat_bias;
DC.s_hat_dither = s_hat_dither;
DC.s_hat_emld = s_hat_emld;
DC.s_hat_mmd = s_hat_mmd;
DC.s_hat_mcd = s_hat_mcd;

DC.cb_idx = cb_idx;
DC.cb_idx_bias = cb_idx_bias;
DC.cb_idx_dither = cb_idx_dither;

DC.yq = yq;
end

