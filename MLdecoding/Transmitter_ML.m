function [x_crc, s_crc] = Transmitter_ML(SP)

Nu = SP.Nu;
M = SP.M;
Nd = SP.Nd;
CRC1 = SP.CRC1;
CRC2 = SP.CRC2;
p = SP.p;
btot = log2(M)*Nd; 
bits = randi([0,1], Nu, btot-CRC2); % data bit sequence - CRC sequence

s_crc = zeros(Nu, Nd);
for u = 1:Nu
    bits_crc = lteCRCEncode(bits(u,:), CRC1); % append CRC
    temp = reshape(bits_crc, log2(M), Nd)';
    sym_crc = bi2de(temp); % symbol index
    s_crc(u,:) = qammod(sym_crc, M, 'UnitAveragePower', true); % QAM modulation
end
x_crc = sqrt(p)*s_crc;

end

