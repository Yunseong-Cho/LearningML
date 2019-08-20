function [H, H_ici] = Channel_Gen_ML(SP)

Nr = SP.Nr;
Nu = SP.Nu;
H_type = SP.H_type;
L = SP.L;
L_ici = SP.L_ici;
switch H_type
    
    case 'Rayleigh'
        H = 1/sqrt(2)*(randn(Nr,Nu) + 1j*randn(Nr,Nu));
        H_ici = 1/sqrt(2)*(randn(Nr,Nu) + 1j*randn(Nr,Nu));
    case 'mmWave'
        H = zeros(Nr,Nu);
        H_ici = zeros(Nr,Nu);
        for u = 1:Nu
            UE(u).L = L; %max(1,poissrnd(L));
            UE(u).g = 1/sqrt(2)*(randn(UE(u).L,1) + 1j*randn(UE(u).L,1));
            UE(u).theta = 2*rand(UE(u).L,1);
            A = SteeringGen(UE(u).theta,Nr);
            H(:,u) = sqrt(Nr/UE(u).L)*A*UE(u).g;
            
            UE_ici(u).L = L_ici; %max(1,poissrnd(L));
            UE_ici(u).g = 1/sqrt(2)*(randn(UE_ici(u).L,1) + 1j*randn(UE_ici(u).L,1));
            UE_ici(u).theta = 2*rand(UE_ici(u).L,1);
            A = SteeringGen(UE_ici(u).theta,Nr);
            H_ici(:,u) = sqrt(Nr/UE_ici(u).L)*A*UE_ici(u).g;
        end
        
end





end

