% Phase 16 reference: geothermal water/brine EOS (MRST geothermal/eos)
%
% Spivey et al. (2004) density and Kestin viscosity, over a grid of
% pressures, temperatures and salinities spanning the correlations' range -
% including points outside it, to pin down the clamping.

mrstModule add geothermal ad-core

here = fileparts(mfilename('fullpath'));

p = [1, 50, 100, 200, 400, 700, 1200, 2500]'*barsa;   % 2500 bar is past the cap
T = [5, 20, 60, 100, 124.9, 125.1, 180, 250, 300]' + 273.15;  % 300 C is past it
c = [0, 0.02, 0.05, 0.10, 0.20, 0.26]';               % NaCl mass fraction

[P, TT, CC] = ndgrid(p, T, c);
P = P(:); TT = TT(:); CC = CC(:);

rho_w = density_pure_water(P, TT);
mu_w  = viscosity_pure_water(P, TT);
rho_b = density_brine(P, TT, CC);
mu_b  = viscosity_brine(P, TT, CC);

ref = struct('p', P, 'T', TT, 'c', CC, ...
             'rho_w', rho_w, 'mu_w', mu_w, ...
             'rho_b', rho_b, 'mu_b', mu_b);
save(fullfile(here, 'ref_p16_geothermal_eos.mat'), '-struct', 'ref', '-v7');

fprintf(['Wrote Phase-16 EOS reference (%d states): rho %.1f-%.1f kg/m3, ' ...
         'mu %.3f-%.3f cP\n'], numel(P), min(rho_b), max(rho_b), ...
        min(mu_b)*1e3, max(mu_b)*1e3);
