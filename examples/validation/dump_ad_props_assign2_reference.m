%% Reference values for the remaining ad-props assign* functions
%
% ref_ad_props_assign.mat covers the core PVT, saturation and EOR keywords.
% This covers the rest that MRST can actually evaluate: the three-column
% saturation family, the solvent and miscibility tables, and the small
% record-valued keywords.
%
% Four are deliberately absent, and each for a stated reason:
%
%   assignWATERVIST, assignOILVIST  index reg.SATKINX, which handleRegions
%                                   never creates - they raise in MRST.
%   assignRelPermScal (3-phase)     passes f where sgmax is expected and uses
%                                   an undefined sg_max - it raises in MRST.
%   assignPLYSHLOG                  stores the record unchanged; there is no
%                                   computed value to compare.

thisDir = fileparts(mfilename('fullpath'));
mrstModule add ad-core ad-props deckformat

reg = struct('PVTNUM', [], 'PVTINX', ':', ...
             'SATNUM', [], 'SATINX', ':', ...
             'ROCKNUM', [], 'ROCKINX', ':', ...
             'SURFNUM', [], 'SURFINX', ':', ...
             'pvt', 1, 'sat', 1, 'prange', [], ...
             'interp1d', @interpTable, 'interp1d_uniform', @interpTable, ...
             'pvtMethodOil', 'parallel', 'pvtMethodGas', 'linshift', ...
             'useMex', false, 'optimize', false);

out = struct();

% ------------------------------------- three-column saturation family
swfn = [0.15, 0.00, 3.0e4; 0.40, 0.10, 8.0e3;
        0.70, 0.45, 1.0e3; 0.85, 1.00, 0.00];
sgfn = [0.00, 0.00, 0.0;   0.10, 0.00, 1.0e3;
        0.40, 0.25, 6.0e3; 0.80, 0.90, 2.0e4];
sof3 = [0.00, 0.00, 0.00; 0.20, 0.00, 0.00;
        0.50, 0.30, 0.25; 0.85, 1.00, 0.90];

s = reshape(linspace(-0.1, 1.1, 25), [], 1);
out.s = s;

f = assignSWFN(struct(), {swfn}, reg);
out.SWFN_krW  = f.krW{1}(s);
out.SWFN_pcOW = f.pcOW{1}(s);
out.SWFN_pts  = f.krPts.w;

f = assignSGFN(f, {sgfn}, reg);
out.SGFN_krG  = f.krG{1}(s);
out.SGFN_pcOG = f.pcOG{1}(s);
out.SGFN_pts  = f.krPts.g;

g = assignSOF3(struct(), {sof3}, reg);
out.SOF3_krOW = g.krOW{1}(s);
out.SOF3_krOG = g.krOG{1}(s);
out.SOF3_pts_ow = g.krPts.ow;
out.SOF3_pts_og = g.krPts.og;

sof2 = sof3(:, [1, 2]);
h = assignSOF2(struct(), {sof2}, reg);
out.SOF2_krO = h.krO(s);

% ------------------------------------------------------- SLGOF -> SGOF
swof = [0.12, 0.00, 1.00, 0.0; 0.90, 1.00, 0.00, 0.0];
slgof = [0.12, 1.00, 0.00, 0.0; 0.50, 0.40, 0.10, 0.0;
         0.80, 0.05, 0.55, 0.0; 1.00, 0.00, 1.00, 0.0];
l = assignSWOF(struct(), {swof}, reg);
l = assignSLGOF(l, {slgof}, reg);
out.SLGOF_krG  = l.krG{1}(s);
out.SLGOF_krOG = l.krOG{1}(s);
out.SLGOF_pts_g = l.krPts.g;

% -------------------------------------------------------------- SGWFN
sgwfn = [0.00, 0.00, 1.00, 0.0; 0.30, 0.10, 0.50, 2.0e3;
         0.70, 0.60, 0.10, 8.0e3; 1.00, 1.00, 0.00, 2.0e4];
w = assignSGWFN(struct(), {sgwfn}, reg);
out.SGWFN_krG  = w.krG(s);
out.SGWFN_krW  = w.krW(s);
out.SGWFN_pcWG = w.pcWG(s);

% --------------------------------------------------- solvent and misc
ssfn = [0.00, 0.00, 1.00; 0.40, 0.35, 0.60; 1.00, 1.00, 0.00];
v = assignSSFN(struct(), {ssfn}, reg);
out.SSFN_krFG = v.krFG(s);
out.SSFN_krFS = v.krFS(s);

msfn = [0.00, 0.00, 1.00; 0.50, 0.45, 0.55; 1.00, 1.00, 0.00];
m = assignMSFN(struct(), {msfn}, reg);
out.MSFN_MkrG = m.MkrG(s);
out.MSFN_MkrO = m.MkrO(s);

p = reshape(linspace(5e6, 5e7, 9), [], 1);
out.p = p;

pvds = [1.0e7, 5.0e-3, 1.4e-5; 2.0e7, 3.0e-3, 1.7e-5; 3.0e7, 2.5e-3, 2.0e-5];
d = assignPVDS(struct(), {pvds}, reg);
out.PVDS_bS  = d.bS(p);
out.PVDS_muS = d.muS(p);

rocktab = [1.0e6, 0.98, 0.90; 2.0e7, 1.00, 1.00; 5.0e7, 1.05, 1.20];
r = assignROCKTAB(struct(), {rocktab}, reg);
out.ROCKTAB_pvMultR   = r.pvMultR(p);
out.ROCKTAB_tranMultR = r.tranMultR(p);

% ------------------------------------------------ small record keywords
sd = assignSDENSITY(struct(), [700.0], reg);
out.SDENSITY_rhoSS = sd.rhoSS;

rc = assignRSCONSTT(struct(), [25.0, 1.5e7], reg);
out.RSCONSTT = [rc.rs, rc.pBub];

eh = assignEHYSTR(struct(), {0.1, 2, 1.0, 0.1, 'BOTH', 'RETR', 'DRAIN', ...
                             'OIL', 'NO', 'NO', 'NO', 0.0, 0}, reg);
out.EHYSTR_flags = [isfield(eh, 'krHyst'), isfield(eh, 'pcHyst')];

% ------------------------------------------------------------ relPerm
% Three-phase dispatch, on a fluid whose curves are already single functions.
rp = assignSWOF(struct(), {[0.10, 0.00, 1.00, 0.0;
                            0.50, 0.20, 0.40, 0.0;
                            0.90, 1.00, 0.00, 0.0]}, reg);
rp = assignSGOF(rp, {[0.00, 0.00, 1.00, 0.0;
                      0.40, 0.30, 0.35, 0.0;
                      0.90, 1.00, 0.00, 0.0]}, reg);
for fld = {'krW', 'krOW', 'krG', 'krOG'}
    rp.(fld{1}) = rp.(fld{1}){1};
end
rp.sWcon = 0.10;
rp = assignRelPerm(rp);

sw = reshape(linspace(0.15, 0.60, 7), [], 1);
sg = reshape(linspace(0.05, 0.30, 7), [], 1);
[krW3, krO3, krG3] = rp.relPerm(sw, sg);
out.relPerm_sw  = sw;
out.relPerm_sg  = sg;
out.relPerm_krW = krW3;
out.relPerm_krO = krO3;
out.relPerm_krG = krG3;

save(fullfile(thisDir, 'ref_ad_props_assign2.mat'), '-struct', 'out', '-v7');
fprintf('Wrote ref_ad_props_assign2.mat with %d fields\n', ...
        numel(fieldnames(out)));
