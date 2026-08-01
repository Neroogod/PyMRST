%% Reference values for the ad-props assign* functions
%
% Each assign* function turns a deck table into property closures. This
% evaluates those closures on fixed probe points so the Python port can be
% compared against MRST rather than against hand-derived expectations - the
% undersaturated PVT branches in particular follow a shifted "parallel"
% interpolation that is easy to reason about incorrectly.

thisDir = fileparts(mfilename('fullpath'));
mrstModule add ad-core ad-props deckformat

out = struct();

% ---------------------------------------------------------------- reg setup
reg = struct();
reg.PVTNUM = [];
reg.PVTINX = ':';
reg.SATNUM = [];
reg.SATINX = ':';
reg.pvt = 1;
reg.sat = 1;
reg.prange = [];
reg.interp1d = @interpTable;
reg.interp1d_uniform = @interpTable;
% initDeckADIFluid's defaults: oil and gas use *different* strategies.
reg.pvtMethodOil = 'parallel';
reg.pvtMethodGas = 'linshift';
reg.useMex = false;
reg.optimize = false;

% ------------------------------------------------------------------- PVTO
pvto = struct('key' , [10; 50], ...
              'pos' , [1; 3; 5], ...
              'data', [1.0e7, 1.20, 1.0e-3;
                       2.0e7, 1.18, 1.1e-3;
                       2.0e7, 1.40, 0.8e-3;
                       3.0e7, 1.37, 0.9e-3]);

f = assignPVTO(struct(), {pvto}, reg);

p_probe  = [1.0e7; 1.5e7; 2.0e7; 2.5e7; 3.0e7];
rs_probe = [10; 10; 50; 50; 50];
flag_sat = logical([1; 0; 1; 0; 0]);

out.PVTO_p    = p_probe;
out.PVTO_rs   = rs_probe;
out.PVTO_flag = double(flag_sat);
out.PVTO_bO   = f.bO{1}(p_probe, rs_probe, flag_sat);
out.PVTO_muO  = f.muO{1}(p_probe, rs_probe, flag_sat);
out.PVTO_rsSat = f.rsSat{1}([0; 5e6; 1.0e7; 1.5e7; 2.0e7; 3.0e7]);
out.PVTO_rsSat_p = [0; 5e6; 1.0e7; 1.5e7; 2.0e7; 3.0e7];
out.PVTO_pb   = f.pb{1}([-0.5; 0; 10; 30; 50]);
out.PVTO_pb_rs = [-0.5; 0; 10; 30; 50];

% ------------------------------------------------------------------- PVTG
pvtg = struct('key' , [1.0e7; 2.0e7], ...
              'pos' , [1; 3; 5], ...
              'data', [1.0e-4, 1.0e-2, 1.4e-5;
                       0.0,    1.1e-2, 1.5e-5;
                       3.0e-4, 5.0e-3, 1.8e-5;
                       0.0,    5.5e-3, 1.9e-5]);

g = assignPVTG(struct(), {pvtg}, reg);

pg_probe = [1.0e7; 1.0e7; 1.5e7; 2.0e7; 2.0e7];
rv_probe = [1.0e-4; 0.0;   0.0;   3.0e-4; 0.0];
gflag    = logical([1; 0; 0; 1; 0]);

out.PVTG_p    = pg_probe;
out.PVTG_rv   = rv_probe;
out.PVTG_flag = double(gflag);
out.PVTG_bG   = g.bG{1}(pg_probe, rv_probe, gflag);
out.PVTG_muG  = g.muG{1}(pg_probe, rv_probe, gflag);
out.PVTG_rvSat_p = [0; 5e6; 1.0e7; 1.5e7; 2.0e7];
out.PVTG_rvSat   = g.rvSat{1}(out.PVTG_rvSat_p);

% ------------------------------------------------------------ simple ones
pvdo = [1.0e7, 1.20, 1.0e-3; 2.0e7, 1.15, 1.1e-3; 3.0e7, 1.10, 1.2e-3];
d = assignPVDO(struct(), {pvdo}, reg);
out.PVDO_p   = [1.0e7; 1.5e7; 2.0e7; 3.0e7];
out.PVDO_bO  = d.bO{1}(out.PVDO_p);
out.PVDO_muO = d.muO{1}(out.PVDO_p);

pvdg = [1.0e7, 5.0e-3, 1.4e-5; 2.0e7, 3.0e-3, 1.7e-5; 3.0e7, 2.5e-3, 2.0e-5];
d = assignPVDG(struct(), {pvdg}, reg);
out.PVDG_p   = [0; 5e6; 1.0e7; 1.5e7; 3.0e7];
out.PVDG_bG  = d.bG{1}(out.PVDG_p);
out.PVDG_muG = d.muG{1}(out.PVDG_p);

pvtw = [2.7680382e7, 1.029, 4.5396812e-10, 3.1e-4, 0.0];
w = assignPVTW(struct(), pvtw, reg);
out.PVTW_p   = [1.0e6; 1.0e7; 2.7680382e7; 5.0e7];
out.PVTW_bW  = w.bW{1}(out.PVTW_p);
out.PVTW_muW = w.muW{1}(out.PVTW_p);

pvcdo = [2.0e7, 1.20, 1.0e-9, 1.0e-3, 2.0e-10];
c = assignPVCDO(struct(), pvcdo, reg);
out.PVCDO_p   = [1.0e7; 2.0e7; 5.0e7];
out.PVCDO_bO  = c.bO{1}(out.PVCDO_p);
out.PVCDO_muO = c.muO{1}(out.PVCDO_p);

rock = [1.0132929e5, 4.3511321e-10];
r = assignROCK(struct(), rock, reg);
out.ROCK_p       = [0; 1.0132929e5; 1.0e7; 3.0e7];
out.ROCK_pvMultR = r.pvMultR(out.ROCK_p);

dens = assignDENSITY(struct(), [49.1, 64.79, 0.06054], reg);
out.DENSITY_rhoOS = dens.rhoOS;
out.DENSITY_rhoWS = dens.rhoWS;
out.DENSITY_rhoGS = dens.rhoGS;

grav = assignGRAVITY(struct(), [45.5, 1.0, 0.7773], reg);
out.GRAVITY_rhoOS = grav.rhoOS;
out.GRAVITY_rhoWS = grav.rhoWS;
out.GRAVITY_rhoGS = grav.rhoGS;

% ------------------------------------------------- saturation functions
swof = [0.12, 0.00, 1.00, 0.0;
        0.30, 0.05, 0.60, 0.0;
        0.60, 0.30, 0.15, 0.0;
        0.90, 1.00, 0.00, 0.0];
sgof = [0.00, 0.00, 1.00, 0.0;
        0.20, 0.05, 0.55, 0.0;
        0.50, 0.40, 0.10, 0.0;
        0.88, 1.00, 0.00, 0.0];

s = assignSWOF(struct(), {swof}, reg);
out.SWOF_sw    = [0.05; 0.12; 0.30; 0.45; 0.90; 1.00];
out.SWOF_krW   = s.krW{1}(out.SWOF_sw);
out.SWOF_so    = [0.10; 0.40; 0.70; 0.88; 0.95];
out.SWOF_krOW  = s.krOW{1}(out.SWOF_so);
out.SWOF_pts_w  = s.krPts.w;
out.SWOF_pts_ow = s.krPts.ow;
out.SWOF_hasPC  = double(isfield(s, 'pcOW'));

s = assignSGOF(s, {sgof}, reg);
out.SGOF_sg    = [0.00; 0.20; 0.35; 0.88; 1.00];
out.SGOF_krG   = s.krG{1}(out.SGOF_sg);
out.SGOF_so    = [0.00; 0.30; 0.68; 0.88];
out.SGOF_krOG  = s.krOG{1}(out.SGOF_so);
out.SGOF_pts_g  = s.krPts.g;
out.SGOF_pts_og = s.krPts.og;
out.SGOF_hasPC  = double(isfield(s, 'pcOG'));

% with a capillary pressure column, so the pc branch is exercised too
swof_pc = swof; swof_pc(:,4) = [3.0e4; 1.0e4; 2.0e3; 0.0];
sp = assignSWOF(struct(), {swof_pc}, reg);
out.SWOF_pc_sw = out.SWOF_sw;
out.SWOF_pcOW  = sp.pcOW{1}(out.SWOF_pc_sw);

% ------------------------------------------------- EOR and thermal (phase 6)
reg.SATNUM = [];  reg.SATINX = ':';
reg.ROCKNUM = []; reg.ROCKINX = ':';
reg.SURFNUM = []; reg.SURFINX = ':';

plyvisc = [0.0, 1.0; 1.0, 5.0; 3.0, 20.0];
p = assignPLYVISC(struct(), {plyvisc}, reg);
out.PLYVISC_c = [-1.0; 0.0; 0.5; 1.0; 3.0; 5.0];
out.PLYVISC_muWMult = p.muWMult{1}(out.PLYVISC_c);

plyads = [0.0, 0.0; 1.0, 0.002; 3.0, 0.003];
p = assignPLYADS(struct(), {plyads}, reg);
out.PLYADS_c = out.PLYVISC_c;
out.PLYADS_ads = p.ads{1}(out.PLYADS_c);

plyshear = [0.0, 1.0; 1.0e-5, 0.9; 1.0e-4, 0.5];
p = assignPLYSHEAR(struct(), {plyshear}, reg);
out.PLYSHEAR_v = [0.0; 5.0e-6; 1.0e-5; 1.0e-4; 1.0e-3];
out.PLYSHEAR_mult = p.plyshearMult{1}(out.PLYSHEAR_v);

surfst = [0.0, 0.05; 1.0, 0.001; 5.0, 1.0e-5];
s = assignSURFST(struct(), {surfst}, reg);
out.SURFST_c = [0.0; 0.5; 1.0; 5.0; 9.0];
out.SURFST_ift = s.ift{1}(out.SURFST_c);

surfvisc = [0.0, 5.0e-4; 1.0, 8.0e-4; 5.0, 2.0e-3];
s = assignSURFVISC(struct(), {surfvisc}, reg);
out.SURFVISC_c = out.SURFST_c;
out.SURFVISC_muWSft = s.muWSft{1}(out.SURFVISC_c);

surfads = [0.0, 0.0; 1.0, 0.0005; 5.0, 0.001];
s = assignSURFADS(struct(), {surfads}, reg);
out.SURFADS_c = out.SURFST_c;
out.SURFADS_ads = s.surfads{1}(out.SURFADS_c);

surfcapd = [-10.0, 0.0; -5.0, 0.5; -2.0, 1.0];
s = assignSURFCAPD(struct(), {surfcapd}, reg);
out.SURFCAPD_nc = [-20.0; -10.0; -7.5; -5.0; -2.0; 0.0];
out.SURFCAPD_miscfact = s.miscfact{1}(out.SURFCAPD_nc);

misc = [0.0, 0.0; 0.5, 0.6; 1.0, 1.0];
m = assignMISC(struct(), {misc}, reg);
out.MISC_sg = [0.0; 0.25; 0.5; 1.0; 1.5];
out.MISC_Ms = m.Ms(out.MISC_sg);

pmisc = [1.0e7, 0.0; 2.0e7, 0.5; 3.0e7, 1.0];
m = assignPMISC(struct(), {pmisc}, reg);
out.PMISC_p = [0.5e7; 1.0e7; 1.5e7; 2.0e7; 4.0e7];
out.PMISC_Mp = m.Mp(out.PMISC_p);

specheat = [300.0, 1000.0, 4180.0, 1900.0;
            350.0, 1100.0, 4200.0, 2000.0;
            400.0, 1200.0, 4260.0, 2100.0];
h = assignSPECHEAT(struct(), {specheat}, reg);
out.SPECHEAT_T = [250.0; 300.0; 325.0; 350.0; 400.0; 450.0];
out.SPECHEAT_uG = h.uG(out.SPECHEAT_T);
out.SPECHEAT_uW = h.uW(out.SPECHEAT_T);
out.SPECHEAT_uO = h.uO(out.SPECHEAT_T);

specrock = [300.0, 2.0e6; 350.0, 2.1e6; 400.0, 2.2e6];
r = assignSPECROCK(struct(), {specrock}, reg);
out.SPECROCK_T = out.SPECHEAT_T;
out.SPECROCK_eR = r.eR(out.SPECROCK_T);

pr = assignPLYROCK(struct(), [0.05, 1.3, 2650.0, 1.0, 2.5e-5], reg);
out.PLYROCK = [pr.dps, pr.rrf, pr.rhoR, pr.adsInx, pr.adsMax];

sr = assignSURFROCK(struct(), [2.0, 2650.0], reg);
out.SURFROCK = [sr.adsInxSft, sr.rhoRSft];

out.PLYMAX_cpmax = getfield(assignPLYMAX(struct(), [3.0, 0.0], reg), 'cpmax');
out.SHRATE_shrate = getfield(assignSHRATE(struct(), [4.8], reg), 'shrate');
out.PLMIXPAR_mixPar = getfield(assignPLMIXPAR(struct(), [1.0], reg), 'mixPar');
tl = assignTLMIXPAR(struct(), [0.7, 0.3], reg);
out.TLMIXPAR = [tl.mixPar, tl.mixParRho];
tl1 = assignTLMIXPAR(struct(), [0.7], reg);
out.TLMIXPAR_one = [tl1.mixPar, tl1.mixParRho];

vr = assignVISCREF(struct(), [2.0e7, 50.0], reg);
out.VISCREF = vr.refVisc;

% WATERVIST and OILVIST are deliberately absent: both index reg.SATKINX,
% a field handleRegions never creates, so they error in MRST as written.

save(fullfile(thisDir, 'ref_ad_props_assign.mat'), '-struct', 'out', '-v7');
fprintf('Wrote ref_ad_props_assign.mat with %d fields\n', ...
        numel(fieldnames(out)));
