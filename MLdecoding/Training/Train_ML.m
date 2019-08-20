function [TR] = Train_ML(i, SP, H, H_ici)

% SP.Nr = SP.Nr(i);
% SP.Nu = SP.Nu(i);
% SP.M = SP.M(i);
SP.p = SP.p(i);
SP.p_dither = SP.p_dither(i);
% SP.p_ici = SP.p_ici(i);
%%

Nr = SP.Nr;
Nu = SP.Nu;
M = SP.M;
Ntr = SP.Ntr;
p = SP.p;
p_dither = SP.p_dither;
p_ici = SP.p_ici;
symbols = {0:M-1};
for u = 1:Nu-1
    symbols = {symbols{:}, 0:M-1}; %cell array with N vectors to combine
end
combinations = cell(1, numel(symbols)); %set up the varargout result
[combinations{:}] = ndgrid(symbols{:});
combinations = cellfun(@(x) x(:), combinations,'uniformoutput',false); %there may be a better way to do this
result = [combinations{:}]; % NumberOfCombinations by N matrix. Each row is unique.
codebook = result';
x = sqrt(p)*qammod(codebook, M, 'UnitAveragePower', true); % Generate symbol

w1 = zeros(2*Nr, M^Nu);
w1d = zeros(2*Nr, M^Nu);
for sym = 1:M^Nu % For all symbols
    yq = zeros(2*Nr,Ntr);
    ydq = zeros(2*Nr,Ntr);
    for i = 1:Ntr % Number of Training
        x_ici = sqrt(p_ici)*qammod(codebook(:,randi([1,M^Nu],1)), M, 'UnitAveragePower', true);
        n = 1/sqrt(2)*(randn(Nr,1) + 1j*randn(Nr,1));
        d = sqrt(p_dither/2)*(randn(Nr,1) + 1j*randn(Nr,1)); % Gaussian dithering
        y = H*x(:,sym) + H_ici*x_ici + n ; % Received signal
        yd = H*x(:,sym) + H_ici*x_ici + n + d; % Received signal + dithering
        
        % (Non-dither) 1-bit Quantization
        tempr = sign(real(y));
        tempi = sign(imag(y));
        yq_real = (tempr == 1);
        yq_imag = (tempi == 1);
        yq(1:Nr,i) = yq_real;
        yq(Nr+1:2*Nr,i) = yq_imag;
        
        % (Dither) 1-bit Quantization
        tempr = sign(real(yd));
        tempi = sign(imag(yd));
        ydq_real = (tempr == 1);
        ydq_imag = (tempi == 1);
        ydq(1:Nr,i) = ydq_real;
        ydq(Nr+1:2*Nr,i) = ydq_imag;
    end
    
    w = sum(yq,2);
    w1(:,sym) = w;
    
    wd = sum(ydq,2);
    w1d(:,sym) = wd;
    
end

idx0 = w1 == 0; % Case: all sequences are 0
idx1 = w1 == Ntr; % Case: all sequences are 1
TR.NonFlip = sum(sum(idx0+idx1))/M^Nu; % Average number of nonflip across all symbols

idx0 = w1d == 0;
idx1 = w1d == Ntr;
TR.NonFlip_dither = sum(sum(idx0+idx1))/M^Nu;

end



