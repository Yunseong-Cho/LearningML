%% Robust Learning-based 1-bit ML Decoding
clear all;

% SYSTEM PARAMETERS
SP.Nr = 32; % Number of antennas
SP.Nu = 4;  % Number of users
SP.M = 4;   % M-QAM
SP.Ntr = 100;   % Numer of training for each symbol vector
SP.SNR_dB = linspace(-10,15,15);    % SNR per antenna y = pHx + n  (snr = p/n)
SP.p = 10.^(SP.SNR_dB/10);  % N0 = 1 (fixed)
SP.H_type = 'Rayleigh'; % Channel type (Rayleigh or mmWave)
SP.L = 4; % mmWave channel paths
SP.p_dither = SP.p/2; % Dithering variance
SP.Num_h = 20;   % Number of block fading channels
SP.D = 1;   % Number of data subframes
SP.Nd = 10;  % Length of data subframe

% MULTI-CELL CONFIG (TWO-CELL)
SP.L_ici = 1; % mmWave channel paths for ICI
SP.p_ici_dB = 5; % Inter-cell-interference power
SP.p_ici = 0 %10.^(SP.p_ici_dB/10);

%%

Nu = SP.Nu;
Num_h = SP.Num_h;
Nd = SP.Nd;
D = SP.D;
Num_case = length(SP.p);

Avg_Est_SNR_dB = zeros(1,Num_case);
Avg_NF = zeros(1,Num_case);
Avg_NF_dither = zeros(1,Num_case);
Avg_CRC_cnt_m1ML = zeros(1,Num_case);
Avg_CRC_cnt_m1ML_dither = zeros(1,Num_case);
for i = 1:Num_case
    tic
  
    NonFlip = zeros(Num_h,1);
    NonFlip_dither = zeros(Num_h,1);
    rng(0)
    for h = 1:Num_h
        [H, H_ici] = Channel_Gen_ML(SP); % Generate channel
        [TR] = Train_ML(i, SP, H, H_ici); % Training phase (generate symbol-hypervectors)
          
        NonFlip(h) = TR.NonFlip;
        NonFlip_dither(h) = TR.NonFlip_dither;
    end
    Avg_NF(i) =  mean(NonFlip);
    Avg_NF_dither(i) =  mean(NonFlip_dither);
 
    disp('===================================================================')
    disp('  NonFlip(bi) NonFlip(di) ')
    disp([ Avg_NF(i), Avg_NF_dither(i)])
   
    toc
end
