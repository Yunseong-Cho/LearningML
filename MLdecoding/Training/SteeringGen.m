function [a] = SteeringGen(theta, NumAntenna)

a = 0:1:NumAntenna-1;
a = (1/sqrt(NumAntenna))*exp(-1j*pi*theta*a);
a = a.';
end

