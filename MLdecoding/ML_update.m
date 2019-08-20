function [TR] = ML_update(SP, TR, Yq, idx, flag)

Nd = SP.Nd;

if flag == 0 % Conventional 
    N = TR.N;
    Num1_updated = TR.Num1;
    for t = 1:Nd % for each symbol
        Num1_updated(:,idx(t)) = Num1_updated(:,idx(t)) + Yq(:,t);
        N(idx(t)) =  N(idx(t)) + 1;
    end
    TR.Num1 = Num1_updated;
    TR.N_1ML = N;
    
elseif  flag == 1 % Bias
    
    N = TR.N_bias;
    Num1_updated = TR.Num1_bias;
    for t = 1:Nd
        Num1_updated(:,idx(t)) = Num1_updated(:,idx(t))+ Yq(:,t);
        N(idx(t)) =  N(idx(t)) + 1;
    end
    TR.Num1_bias = Num1_updated;
    TR.N_bias = N;
    
elseif  flag == 2 % Dither
    
    N = TR.N_dither;
    Num1_updated = TR.Num1_dither_post;
    for t = 1:Nd
        Num1_updated(:,idx(t)) = Num1_updated(:,idx(t))+ Yq(:,t);
        N(idx(t)) =  N(idx(t)) + 1;
    end
    TR.Num1_dither_post = Num1_updated;
    TR.N_dither = N;
end

