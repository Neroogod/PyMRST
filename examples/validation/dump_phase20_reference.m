% Phase 20 reference: upscaled VE relperm/pc with a capillary fringe
% (co2lab-ve/fluids, addVERelpermCapillaryFringe 'P-scaled table' + free_sg)
%
% NB the 'S table' branch of addVERelpermCapillaryFringe cannot be dumped: its
% truncate_at_aquifer_bottom indexes a misspelled variable ('trunctate') and
% raises. Only the (default) P-scaled branch is exercised here.

mrstModule add co2lab-common co2lab-ve ad-core ad-props
gravity reset on

here = fileparts(mfilename('fullpath'));

nc = 6;
H  = [10; 20; 20; 30; 20; 50];
G  = computeGeometry(cartGrid([nc 1 1], [nc*100, 100, 20]));
Gt = topSurfaceGrid(G);
Gt.cells.H = H;                       % vary the thickness cell by cell

rock2D = struct('poro', repmat(0.2, nc, 1), ...
                'perm', repmat(100*milli*darcy, nc, 1));

srw = 0.2; srg = 0.2;
rhoWS = 1000; rhoGS = 700;
fluid = struct('rhoWS', rhoWS, 'rhoGS', rhoGS, ...
               'res_water', srw, 'res_gas', srg, ...
               'bW', @(p) 1 + 0*p, 'bG', @(p) 1 + 0*p, ...
               'rhoW', @(p) rhoWS + 0*p, 'rhoG', @(p) rhoGS + 0*p, ...
               'muW', @(p) 1e-3 + 0*p, 'muG', @(p) 6e-5 + 0*p);

% a Brooks-Corey-ish fine-scale rock: linear in p up to the entry pressure
entry = 1e5;
invPc3D = @(p) 1 - min(max(p./entry, 0), 1) .* (1 - srw);
kr3D    = @(s) s.^2;

f = addVERelpermCapillaryFringe(fluid, Gt, rock2D, invPc3D, kr3D, ...
                                'type', 'P-scaled table', 'samples', 2000);

p  = repmat(100*barsa, nc, 1);
sg    = [0.02; 0.10; 0.25; 0.40; 0.55; 0.70];
sGmax = [0.02; 0.10; 0.40; 0.40; 0.80; 0.70];   % mix of drainage & imbibition

krG  = f.krG(sg, p, 'sGmax', sGmax);
pcWG = f.pcWG(sg, p, 'sGmax', sGmax);
krW  = f.krW(1 - sg, p, 'sGmax', sGmax);
fsg  = free_sg(sg, sGmax, srw, srg);

% The raw upscaled table, so the integration can be compared on its own.
% Pmax must be what determine_upper_cap_press_limit computes - a doubling
% search, *not* the entry pressure - or the table is sampled differently from
% the one the fluid actually uses and everything drifts by ~1e-5.
dp = 1;
while invPc3D(dp) > srw + 1e-3 && dp < 2e6
    dp = dp * 2;
end
Pmax = dp + max(H) * (rhoWS - rhoGS) * norm(gravity);
samples = 2000;
pp = linspace(0, Pmax, samples);
dp = pp(2) - pp(1);
sw_p = invPc3D(pp); sg_p = 1 - sw_p; kr_p = kr3D(sg_p);
last_zero = find(sg_p == 0, 1, 'last');
if last_zero > 1
    sg_p(1:last_zero) = linspace(0, 0.1*sg_p(last_zero+1), last_zero);
end
SP  = (cumsum(sg_p) - sg_p/2) * dp;
krP = (cumsum(kr_p) - kr_p/2) * dp;

ref = struct('H', H, 'sg', sg, 'sGmax', sGmax, 'p', p, ...
             'krG', krG, 'pcWG', pcWG, 'krW', krW, 'fsg', fsg, ...
             'srw', srw, 'srg', srg, 'entry', entry, 'Pmax', Pmax, ...
             'tab_p', pp(:), 'tab_SP', SP(:), 'tab_krP', krP(:));
save(fullfile(here, 'ref_p20_ve_fluids.mat'), '-struct', 'ref', '-v7');

fprintf('Wrote Phase-20 reference: krG %s\n', mat2str(round(krG(:)', 5)));
