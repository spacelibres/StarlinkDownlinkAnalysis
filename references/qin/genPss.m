function [pkVec] = genPss()
% genPss : Generates the Starlink downlink primary synchronization sequence
%          (PSS).
%           
%
% OUTPUTS
%
% pkVec ------ (N + Ng)-by-1 PSS
%
%+==============================================================================+%

% Duration of OFDM symbol guard interval (cyclic prefix), expressed in
% intervals of 1/Fs.  
Ng = 32;
% Number of sub-segments in the primary synchronization sequence (PSS)
NpssSeg = 8;

% Fibonacci LFSR
ciVec = [3 7]';
a0Vec = [0 0 1 1 0 1 0]';
n = 7;
m = 2^n - 1;
lfsrSeq = zeros(m,1);
for idx=1:m
  buffer = a0Vec(ciVec);
  % Cascaded XOR is the same as the parity
  val = rem(sum(buffer),2); 
  a0Vec = [val; a0Vec(1:end-1)];
  lfsrSeq(idx) = val;
end
lfsrSeqMod = flipud(lfsrSeq);
lfsrSeqMod = [0; lfsrSeqMod];
seq = 2*lfsrSeqMod-1;
pkSegVec = [exp(-1j*pi/4 - 1j*0.5*pi*cumsum(seq))];
pkVec = [-pkSegVec; repmat(pkSegVec,NpssSeg - 1,1)];
pkCP = pkVec(end-Ng+1:end);
pkVec = [-pkCP; pkVec];
