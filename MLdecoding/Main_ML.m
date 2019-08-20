%% Robust Learning-based 1-bit ML Detection
clear all;

% SNR TRAINING
addpath('TrainingData')
load('trData30_half.mat');            % Training data
SP.polyOrder = 5;                     % Linear regression order
SP.polyFit = SNRtraining(SP, Avg_NF_dither); % Linear regression coefficients

% SYSTEM PARAMETERS
SP.Nr = 32;     % Number of antennas
SP.Nu = 4;      % Number of users
SP.M = 4;       % M-QAM
SP.Ntr = 30;    % Numer of training for each symbol vector
SP.bias  = 1/100/SP.Ntr;        % Bias probability
SP.SNR_dB = linspace(-8,8,5);   % SNR y = pHx + n  (snr = p/N0)
SP.p = 10.^(SP.SNR_dB/10);      % N0 = 1 (fixed)
SP.H_type = 'Rayleigh';         % Channel type (Rayleigh or mmWave)
SP.CDF = 'approx';              % CDF type (exact vs. approx)
SP.L = 4;               % mmWave channel paths
SP.p_dither = SP.p/2;   % Dithering variance ~ CN(0, p_dither)
SP.CRC1 = '16';         % CRC length in string
SP.CRC2 = 16;           % CRC length
SP.Num_h = 10;          % Number of block fading channel realizations
SP.D = 1;               % Number of data subframes
SP.Nd = 200;            % Length of data subframe (Data frame = D x Nd)
SP.a_update = 1 - linspace(0.01,0.5,SP.D*SP.Nd); % post update weight for dithering approach

%%

Nu = SP.Nu;
Num_h = SP.Num_h;
Nd = SP.Nd;
D = SP.D;
Num_case = length(SP.p);

SER_opt = zeros(Num_case,1);    % Optimal 1-bit ML
SER_ZF = zeros(Num_case,1);     % 1-bit ZF
SER = zeros(Num_case,1);        % Conventional learning-based 1-bit ML
SER_bias = zeros(Num_case,1);   % Biased-learning 1-bit ML (proposed)
SER_dither = zeros(Num_case,1); % Dithering-and-learning 1-bit ML (proposed)
SER_eMLD = zeros(Num_case,1);   % empherical ML
SER_MMD = zeros(Num_case,1);    % Minimum-Mean Distance
SER_MCD = zeros(Num_case,1);    % Minimum-Center Distance

Avg_Est_SNR_dB = zeros(1,Num_case);
Avg_NF = zeros(1,Num_case);                 % Average number of no sign flip (out of 2xNr)
Avg_NF_dither = zeros(1,Num_case);          % Average number of no sign flip (out of 2xNr) for dithering
Avg_CRC_cnt_bias = zeros(1,Num_case);       % Average number of correctly decoded data subframes
Avg_CRC_cnt_dither = zeros(1,Num_case);     % Average number of correctly decoded data subframes
for i = 1:Num_case
    tic
    error_count_opt = 0;
    error_count_zf = 0;
    error_count = 0;
    error_count_bias = 0;
    error_count_dither = 0;
    error_count_emld = 0;
    error_count_mmd = 0;
    error_count_mcd = 0;
    
    CRC_cnt_bias = zeros(Num_h,1);
    CRC_cnt_dither = zeros(Num_h,1);
    
    Est_SNR_dB = zeros(Num_h,1);
    NF = zeros(Num_h,1);
    NF_dither = zeros(Num_h,1);
    
    rng(1)
    for h = 1:Num_h
        [H] = Channel_Gen_ML(SP);     % Generate channel
        [TR] = Train_ML(i, SP, H);    % Training phase (estimate:Likelihood functions, SNR)
        [DT] = Data_ML(i, SP, H, TR); % Data transmission phase
        
        error_count_opt = error_count_opt + DT.error_opt;
        error_count_zf = error_count_zf + DT.error_zf;
        error_count = error_count + DT.error;
        error_count_bias = error_count_bias + DT.error_bias;
        error_count_dither = error_count_dither + DT.error_dither;
        
        error_count_emld = error_count_emld + DT.error_emld;
        error_count_mmd = error_count_mmd + DT.error_mmd;
        error_count_mcd = error_count_mcd + DT.error_mcd;
        
        CRC_cnt_bias(h) = DT.cnt_crc_bias;
        CRC_cnt_dither(h) = DT.cnt_crc_dither;
        
        Est_SNR_dB(h) = TR.est_snr_dB;
        NF(h) = TR.NF;
        NF_dither(h) = TR.NF_dither;
    end
    
    SER_opt(i) = error_count_opt/(Num_h*Nd*Nu*D);
    SER_ZF(i) = error_count_zf/(Num_h*Nd*Nu*D);
    SER(i) = error_count/(Num_h*Nd*Nu*D);
    SER_bias(i) = error_count_bias/(Num_h*Nd*Nu*D);
    SER_dither(i) = error_count_dither/(Num_h*Nd*Nu*D);
    SER_eMLD(i) = error_count_emld/(Num_h*Nd*Nu*D);
    SER_MMD(i) = error_count_mmd/(Num_h*Nd*Nu*D);
    SER_MCD(i) = error_count_mcd/(Num_h*Nd*Nu*D);
    
    Avg_Est_SNR_dB(i) = 10*log10(mean(10.^(Est_SNR_dB/10)));
    Avg_NF(i) =  mean(NF);
    Avg_NF_dither(i) =  mean(NF_dither);
    Avg_CRC_cnt_bias(i) = mean(CRC_cnt_bias);
    Avg_CRC_cnt_dither(i) = mean(CRC_cnt_dither);
    
    disp('===================================================================')
    disp('  Est.SNR | NonFlip(bi) NonFlip(di)  CRC(bi)    CRC(di)')
    disp([ Avg_Est_SNR_dB(i), Avg_NF(i),Avg_NF_dither(i),Avg_CRC_cnt_bias(i), Avg_CRC_cnt_dither(i)])
    disp('     SNR  |   SER(bi)   SER(di)  SER(naive) SER(opt)  SER(ZF) SER(eMLD)   SER(MMD)    SER(MCD)')
    disp([SP.SNR_dB(i), SER_bias(i), SER_dither(i), SER(i), SER_opt(i), SER_ZF(i), SER_eMLD(i), SER_MMD(i), SER_MCD(i)])
    
    toc
end

%%

n = SP.SNR_dB;
figure
semilogy(n, SER, 'bo-', n, SER_ZF,'g-s',n, SER_bias, 'r^-', n, SER_dither, 'r^--',n, SER_eMLD, 'bo-' ,n, SER_MMD, 'b^-' ,n, SER_MCD, 'b*-' ,n, SER_opt,'k-x')
legend('Learning 1-bit ML', '1-bit ZF', 'Biased Learning 1-bit ML', 'Dithered Learning 1-bit ML', 'empirical MLD(eMLD)','Minimum-Mean-Distance (MMD)','Minimum-Center-Distance (MCD)','Optimal 1-bit ML')
xlabel('SNR $\gamma$ [dB]','interpret','latex')
ylabel('Symbol Error Rate','interpret','latex')
grid on

