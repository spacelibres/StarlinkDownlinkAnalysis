function [templateFrame, rotations] = genReferenceTemplate()
% genReferenceTemplate : Generates the Starlink Ku-band downlink reference
%                        template -- an all-QPSK reference frame used for
%                        signal identification, synchronization, and channel
%                        estimation.  The template rotations are the consensus
%                        QPSK pattern extracted from the all-QPSK frames in the
%                        exemplar dataset (referenceTemplate.mat); edge-pilot
%                        and gutter subcarriers are excluded.
%
% OUTPUTS
%
% templateFrame ---- Column vector of time-domain IQ samples for the reference
%                    frame, with a length-Ng cyclic prefix prepended to each
%                    OFDM symbol.
%
% rotations -------- N-by-Nsym matrix of frequency-domain QPSK rotation indices
%                    (0, 1, 2, 3 for the four QPSK phases; -1 marks null and
%                    edge-pilot subcarriers).
%
%+==============================================================================+%

starlinkConstants

% Load reference template rotations from .mat file (generated from exemplar frames analysis).
data = load('referenceTemplate.mat', 'referenceTemplateRotations');
rotations = data.referenceTemplateRotations;

% Zero out edge pilot subcarriers (they have their own 4QAM pattern).
% Edge pilots: k = 488:495 and 528:535 (0-indexed), or 489:496 and 529:536 (1-indexed).
gutterIdx    = [1 2 1023 1024];
edgePilotIdx = [489:496, 529:536];
rotations([gutterIdx edgePilotIdx], :) = -1;

templateIQ = exp(1j * rotations * pi/2);
templateIQ(rotations==-1) = 0;

templateFrame = sqrt(sl.N) * ifft(templateIQ);
templateFrame = [templateFrame(end-sl.Ng+1:end,:); templateFrame];
templateFrame = reshape(templateFrame, [numel(templateFrame) 1]);

end
