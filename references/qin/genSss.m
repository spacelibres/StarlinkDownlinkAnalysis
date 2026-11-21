function [sssTimeDomainVec, sssFreqDomainVec] = genSss()
% genSss : Generates the Starlink downlink secondary synchronization
%          sequence (SSS).
%
%
% OUTPUTS
%
% sssTimeDomainVec ----- (N + Ng)-by-1 SSS expressed in the time domain with a
%                        length-Ng cyclic prefix, scaled such that the norm
%                        squared of the N-by-1 time domain vector before
%                        cyclic prefix pre-pending is N - 4.  Such scaling,
%                        which accounts for the absent modulation in the
%                        4-subcarrier mid-channel gutter, ensures that the
%                        magnitude of the SSS produced by this function is
%                        commensurate with that of the the PSS produced by
%                        genPss.
%
% sssFreqDomainVec ----- N-by-1 SSS expressed in the frequency domain with
%                        QPSK symbols (aligned along real and imaginary axes)
%                        having unit coordinates.
%
%+==============================================================================+%

% Number of subcarriers in bandwidth Fs, size of FFT/IFFT, number of
% samples at rate Fs in the useful symbol interval. 
N = 1024;
% Duration of OFDM symbol guard interval (cyclic prefix), expressed in
% intervals of 1/Fs.  
Ng = 32;

% Run estimateSSS.m first
load sssVec;
if(length(sssVecFull) ~= N)
  error('Frequency domain SSS expression must be of length N');
end
if(~isempty(find(abs(sssVecFull(3:end-2)) ~= 1)))
  error('sssVecFull expected to have QPSK symbols with unit magnitudes');
end
sssFreqDomainVec = sssVecFull;
sssTimeDomainVec = (N/sqrt(N))*ifft(sssFreqDomainVec);
sssTimeDomainVec = [sssTimeDomainVec(end-Ng+1:end); sssTimeDomainVec];

