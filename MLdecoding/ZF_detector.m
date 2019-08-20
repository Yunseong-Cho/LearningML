function [s_hat] = ZF_detector(SP, H, y)

Nu = SP.Nu;
M = SP.M;

yq = sign(real(y)) + 1j*sign(imag(y));
x_hat = pinv(H)*yq;
x_hat = sqrt(Nu)*x_hat/norm(x_hat);
s = qammod(0:M-1,M,'UnitAveragePower',true);
s_hat = zeros(Nu,1);
for u = 1:Nu
    [~,idx] = min(abs(x_hat(u) - s));
    s_hat(u,1) = qammod(idx-1,M,'UnitAveragePower',true);
end

end

