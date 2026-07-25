%% Dump Phase 366 reference: grdecl2Rock (Phase 25), from MRST.
%
% Run in MATLAB after starting MRST. Writes ref_p366_grdecl2rock.mat.
%
% Pins the PyMRST `grdecl2rock` port against MRST's `grdecl2Rock` across the
% three permeability layouts: isotropic (lone PERMX), diagonal (PERMX/Y/Z) and
% full symmetric (adding PERMXY/XZ/YZ), plus PORO and NTG. Deterministic inputs,
% so the comparison is exact.

thisDir = fileparts(mfilename('fullpath'));

nc = 5;
base = (1:nc)';

% -- isotropic ---------------------------------------------------------------
gi.PERMX = 10 + base;
gi.PORO  = 0.1 + 0.01*base;
gi.NTG   = 0.9 - 0.02*base;
ri = grdecl2Rock(gi);
perm_iso = ri.perm;   poro_iso = ri.poro;   ntg_iso = ri.ntg;

% -- diagonal ----------------------------------------------------------------
gd.PERMX = 10 + base;
gd.PERMY = 20 + base;
gd.PERMZ = 1 + 0.1*base;
gd.PORO  = 0.2 + 0.01*base;
rd = grdecl2Rock(gd);
perm_diag = rd.perm;

% -- full symmetric ----------------------------------------------------------
gf.PERMX  = 10 + base;
gf.PERMY  = 20 + base;
gf.PERMZ  = 1 + 0.1*base;
gf.PERMXY = 0.5 + 0.01*base;
gf.PERMXZ = 0.3 + 0.01*base;
gf.PERMYZ = 0.2 + 0.01*base;
rf = grdecl2Rock(gf);
perm_full = rf.perm;

save(fullfile(thisDir, 'ref_p366_grdecl2rock.mat'), '-v7', ...
     'perm_iso', 'poro_iso', 'ntg_iso', 'perm_diag', 'perm_full');

fprintf('Wrote ref_p366_grdecl2rock.mat to %s\n', thisDir);
fprintf('  iso cols = %d, diag cols = %d, full cols = %d\n', ...
        size(perm_iso, 2), size(perm_diag, 2), size(perm_full, 2));
