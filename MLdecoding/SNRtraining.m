function [polyFit] = SNRtraining(SP, tData)

polyOrder = SP.polyOrder;
snr_dB = SP.SNR_dB;

polyFit = polyfit(tData, snr_dB, polyOrder);
polyEval = polyval(polyFit, tData);
% expFit = fit(P_total, P_sw, 'exp1');

figure
hold on
plot(tData, snr_dB, 'ko-')
plot(tData, polyEval, 'rx-')
grid on
xlabel('Number of zero likelihood functions')
ylabel('SNR \gamma [dB]')
legend('Training data','Linear regression') 
end

