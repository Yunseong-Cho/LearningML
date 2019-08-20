function [H] = Channel_Gen_ML(SP)

Nr = SP.Nr;
Nu = SP.Nu;
H_type = SP.H_type;
L = SP.L;
switch H_type
    
    case 'Rayleigh'
        H = 1/sqrt(2)*(randn(Nr,Nu) + 1j*randn(Nr,Nu));
        
    case 'mmWave'
        H = zeros(Nr,Nu);
        for u = 1:Nu
            UE(u).L = L; % max(1,poissrnd(L));
            UE(u).g = 1/sqrt(2)*(randn(UE(u).L,1) + 1j*randn(UE(u).L,1));
            UE(u).theta = 2*rand(UE(u).L,1);
            A = SteeringGen(UE(u).theta,Nr);
            H(:,u) = sqrt(Nr/UE(u).L)*A*UE(u).g;
            
        end       
end

end

