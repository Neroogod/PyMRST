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

% -- PERMH and long/reverse aliases ------------------------------------------
gh.PERMH = 30 + base;
gh.PERMZ = 2 + base;
rh = grdecl2Rock(gh);
perm_permh = rh.perm;

ga.PERMXX = 10 + base; ga.PERMYY = 20 + base; ga.PERMZZ = 30 + base;
ga.PERMYX = 1 + 0.1*base; ga.PERMZX = 2 + 0.1*base; ga.PERMZY = 3 + 0.1*base;
ra = grdecl2Rock(ga);
perm_alias = ra.perm;

% Both spellings present: later reverse aliases win in MRST's tensorValues.
gp = gf;
gp.PERMYX = 4 + 0.1*base;
gp.PERMZX = 5 + 0.1*base;
gp.PERMZY = 6 + 0.1*base;
rp = grdecl2Rock(gp);
perm_precedence = rp.perm;

% -- active map and ancillary properties ------------------------------------
gx.PERMX = 100 + base;
gx.PORO = 0.1 + 0.01*base;
gx.NTG = 0.9 - 0.02*base;
gx.ROCKTYPE = base;
gx.MULTX = 10 + base;
gx.MULTX_ = 20 + base;
gx.MULTY = 30 + base;
gx.MULTZ_ = 40 + base;
rx = grdecl2Rock(gx, [1; 3; 5]);
active_perm = rx.perm; active_poro = rx.poro; active_ntg = rx.ntg;
active_rocktype = rx.rocktype;
mult_x = rx.multipliers.x; mult_x_ = rx.multipliers.x_;
mult_y = rx.multipliers.y; mult_z_ = rx.multipliers.z_;

save(fullfile(thisDir, 'ref_p366_grdecl2rock.mat'), '-v7', ...
     'perm_iso', 'poro_iso', 'ntg_iso', 'perm_diag', 'perm_full', ...
     'perm_permh', 'perm_alias', 'perm_precedence', ...
     'active_perm', 'active_poro', 'active_ntg', 'active_rocktype', ...
     'mult_x', 'mult_x_', 'mult_y', 'mult_z_');

fprintf('Wrote ref_p366_grdecl2rock.mat to %s\n', thisDir);
fprintf('  iso cols = %d, diag cols = %d, full cols = %d\n', ...
        size(perm_iso, 2), size(perm_diag, 2), size(perm_full, 2));
