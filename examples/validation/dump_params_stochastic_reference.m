%% Deterministic stochastic-rock algorithm references.

thisDir = fileparts(mfilename('fullpath'));

%% 3-D gaussianField body on a supplied deterministic draw.
dimsG = [8, 7, 5]; szG = [3, 5, 3]; stdG = 1.4; valsG = [0.2, 0.7];
drawG = reshape(sin((1:prod(dimsG))/7), dimsG);
padSz = (szG-1)/2;
[x,y,z] = meshgrid(-padSz(2):padSz(2), -padSz(1):padSz(1), -padSz(3):padSz(3));
D = exp(-(x.*x+y.*y+z.*z)/(2*stdG*stdG)); D = D/sum(D(:));
idx = cell(3,1);
for k=1:3
    M=size(drawG,k); I=ones(1,padSz(k)); idx{k}=[I 1:M M*I];
end
gaussian3 = convn(drawG(idx{:}), D, 'valid');
lo=min(gaussian3(:)); hi=max(gaussian3(:));
gaussian3 = valsG(1)+(valsG(2)-valsG(1))*(gaussian3-lo)/(hi-lo);

%% Uneven mean-controlled logNormLayers body on supplied draws.
dimsL = [9, 8, 5]; boundsL = [1, 3, 6]; meansL = [10, 100];
sigmaL=1.3; aL=.4; bL=1.7; szL=[5,3,3]; stdL=2.2;
drawL1 = reshape(cos((1:(9*8*4))/9), [9,8,4]); shiftL1=.25;
drawL2 = reshape(sin((1:(9*8*5))/11), [9,8,5]); shiftL2=-.3;
logLayers = zeros(dimsL);
draws = {drawL1, drawL2}; shifts = [shiftL1, shiftL2];
for i=1:2
    n=boundsL(i+1)-boundsL(i);
    k=smooth3(draws{i}-aL*shifts(i), 'gaussian', szL, stdL);
    k=exp(bL+sigmaL*k); k=k(:,:,2:n+1);
    logLayers(:,:,boundsL(i):boundsL(i+1)-1)=meansL(i)*k/mean(k(:));
end
logLayers = logLayers(:);

%% Native MRST layer-count structures (values intentionally not compared).
[~, Ldefault] = logNormLayers([4,4,5]);
[~, Lall] = logNormLayers([4,4,5], 5);
[~, Lthree] = logNormLayers([4,4,5], 3);

save(fullfile(thisDir, 'ref_params_stochastic.mat'), '-v7', ...
    'dimsG','szG','stdG','valsG','drawG','gaussian3', ...
    'dimsL','boundsL','meansL','sigmaL','aL','bL','szL','stdL', ...
    'drawL1','drawL2','shiftL1','shiftL2','logLayers', ...
    'Ldefault','Lall','Lthree');
fprintf('Wrote stochastic rock reference to %s\n', thisDir);
