%% MRST reference for initWellSolAD.

thisDir = fileparts(mfilename('fullpath'));
run(fullfile(thisDir, 'dump_ad_core_raw_deck_schedule_reference.m'));
state0 = struct('pressure', [200; 190; 180; 170]*barsa);
W = schedule.control(1).W;
wellSol = initWellSolAD(W, model, state0);

names = {wellSol.name};
status = [wellSol.status];
types = {wellSol.type};
vals = [wellSol.val];
signs = [wellSol.sign];
bhp = [wellSol.bhp];
qWs = [wellSol.qWs];
qOs = [wellSol.qOs];
qGs = [wellSol.qGs];
mixs = {wellSol.mixs};
cstatus = {wellSol.cstatus};
cdp = {wellSol.cdp};
cqs = {wellSol.cqs};
save(fullfile(thisDir, 'ref_ad_core_init_well_sol.mat'), '-v7', ...
    'names', 'status', 'types', 'vals', 'signs', 'bhp', ...
    'qWs', 'qOs', 'qGs', 'mixs', 'cstatus', 'cdp', 'cqs');
fprintf('Wrote ref_ad_core_init_well_sol.mat\n');
