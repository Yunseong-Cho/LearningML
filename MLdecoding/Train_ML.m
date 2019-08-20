function [TR] = Train_ML(i, SP, H)

% SP.Nr = SP.Nr(i);
% SP.Nu = SP.Nu(i);
% SP.M = SP.M(i);
SP.p = SP.p(i);
SP.p_dither = SP.p_dither(i);

Nr = SP.Nr;
Nu = SP.Nu;
M = SP.M;
Ntr = SP.Ntr;
p = SP.p;
p_dither = SP.p_dither;
polyFit = SP.polyFit;

% Codebook Generation (there may be a better way to do this)
symbols = {0:M-1};
for u = 1:Nu-1
    symbols = {symbols{:}, 0:M-1}; % cell array with N vectors to combine
end
combinations = cell(1, numel(symbols)); % set up the varargout result
[combinations{:}] = ndgrid(symbols{:});
combinations = cellfun(@(x) x(:), combinations,'uniformoutput',false);
result = [combinations{:}]; % NumberOfCombinations by N matrix. Each row is unique.
codebook = result'; % each column: codeword

TR.possible_vector = [];
TR.pmf = [];
TR.N_vec = [];

TR.centroid = zeros(2*Nr, M^Nu);
x = sqrt(p)*qammod(codebook, M, 'UnitAveragePower', true); % Generate symbol

Num1 = zeros(2*Nr, M^Nu);
Num1d = zeros(2*Nr, M^Nu);
for sym = 1:M^Nu % For all symbols
    yq = zeros(2*Nr, Ntr);
    ydq = zeros(2*Nr, Ntr);
    for i = 1:Ntr % Number of Training
        n = 1/sqrt(2)*(randn(Nr,1) + 1j*randn(Nr,1)); % AWGN
        d = sqrt(p_dither/2)*(randn(Nr,1) + 1j*randn(Nr,1)); % Gaussian dithering
        y = H*x(:,sym) + n; % Received signal (also used for SL)
        yd = H*x(:,sym) + n + d; % Received signal + dithering
        
        % 1-bit Quantization
        tempr = sign(real(y));
        tempi = sign(imag(y));
        yq_real = (tempr == 1);
        yq_imag = (tempi == 1);
        yq(1:Nr,i) = yq_real;
        yq(Nr+1:2*Nr,i) = yq_imag;
        
        %possible_output(sym,:,i) = num2str(yq(:,i)'); % sym * 2Nr * Ntr
        %centroid(:,sym) == centroid(:,sym) + yq(:,i)/Ntr;
                
        tempr = sign(real(yd));
        tempi = sign(imag(yd));
        ydq_real = (tempr == 1);
        ydq_imag = (tempi == 1);
        ydq(1:Nr,i) = ydq_real;
        ydq(Nr+1:2*Nr,i) = ydq_imag;
        
        
        if(i==1)
               TR.possible_vector(sym,:,i) = yq(:,i);
               TR.pmf(sym, i) = 1/Ntr;     
               TR.N_vec(sym) = 1;
        else       
            for j = 1:TR.N_vec(sym)
                if(isequal(yq(:,i)', TR.possible_vector(sym,:,j))) % current yq already exists in possible_vector
                    TR.pmf(sym, j) = TR.pmf(sym, j) + 1/Ntr;                       % don't generate new vector, but just accumulate the prob.
                    break;
                elseif(j==TR.N_vec(sym) && ~isequal(yq(:,i)', TR.possible_vector(sym,:,j))) % if it cannot find overlapped vector until last step
                    TR.N_vec(sym) = TR.N_vec(sym) + 1;                        % increase the number of possive vector
                    TR.possible_vector(sym,:,TR.N_vec(sym)) = yq(:,i);        % , and generate add new vector
                    TR.pmf(sym, TR.N_vec(sym)) = 1/Ntr;                       % also plug 1/Ntr as pmf since it's new vector
                end  
            end     
        end       
    end
    
    
%     for j = 1:TR.N_vec(sym)
%         TR.centroid(:,sym) = TR.centroid(:,sym) + TR.possible_vector(sym,:,j)'*TR.pmf(sym, j);
%     end
    
    TR.centroid(:,sym) = sum(yq,2)/Ntr;
    
    
    Num1(:,sym) = sum(yq,2); % Number of 1s out of Ntr
    Num1d(:,sym) = sum(ydq,2);
    
end

idx0 = Num1 == 0;
idx1 = Num1 == Ntr;
TR.NF = sum(sum(idx0+idx1))/M^Nu; % non-flip count

idx0 = Num1d == 0;
idx1 = Num1d == Ntr;
TR.NF_dither = sum(sum(idx0+idx1))/M^Nu;

TR.est_snr_dB = polyval(polyFit, TR.NF_dither);
TR.codebook = codebook;

TR.Num1 = Num1;
TR.Num1_bias = Num1;
TR.Num1_dither = Num1d;

% For post update
TR.N = Ntr*ones(1,M^Nu);
TR.N_bias = Ntr*ones(1,M^Nu);
TR.N_dither = zeros(1,M^Nu);
TR.Num1_dither_post = zeros(2*Nr, M^Nu);
end



