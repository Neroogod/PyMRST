%% Fluid properties built by MRST from whole decks
%
% The per-keyword references in ref_ad_props_assign.mat feed each assign*
% function a fixture built by hand. This instead runs MRST's own pipeline end
% to end - readEclipseDeck, convertDeckUnits, initDeckADIFluid - on the SPE
% benchmark decks, and evaluates the resulting fluid.
%
% That closes the last gap: it tests the reader, the unit conversion, the
% region handling and the assignment together, on real deck text, against the
% reference implementation. A defect that cancels between two stages shows up
% here and nowhere else.

thisDir = fileparts(mfilename('fullpath'));
mrstModule add ad-core ad-props deckformat

decks = {'SPE1', 'SPE3', 'SPE9'};
out = struct();

for k = 1 : numel(decks)
   name = decks{k};
   fn   = fullfile(ROOTDIR, 'examples', 'data', name, ['BENCH_', name, '.DATA']);

   deck  = convertDeckUnits(readEclipseDeck(fn));
   fluid = initDeckADIFluid(deck);

   e = struct();
   e.fields = sort(reshape(fieldnames(fluid), 1, []));

   % Pressure probes spanning each deck's own tabulated range.
   p = reshape(linspace(5e6, 5e7, 9), [], 1);
   e.p = p;

   e.bW   = fluid.bW(p);
   e.muW  = fluid.muW(p);
   e.pvMultR = fluid.pvMultR(p);

   e.rhoWS = fluid.rhoWS;
   e.rhoOS = fluid.rhoOS;
   e.rhoGS = fluid.rhoGS;

   % Oil: live (PVTO) or dead (PVDO). The call signatures differ.
   if isfield(fluid, 'rsSat')
      e.live_oil = 1;
      rs = fluid.rsSat(p);
      e.rsSat = rs;
      e.pb    = fluid.pb(rs);
      e.bO    = fluid.bO(p, rs, true(size(p)));
      e.muO   = fluid.muO(p, rs, true(size(p)));
      % undersaturated: hold Rs at half its saturated value
      e.bO_usat  = fluid.bO(p, rs/2, false(size(p)));
      e.muO_usat = fluid.muO(p, rs/2, false(size(p)));
   else
      e.live_oil = 0;
      e.bO  = fluid.bO(p);
      e.muO = fluid.muO(p);
   end

   % Gas: wet (PVTG) or dry (PVDG).
   if isfield(fluid, 'rvSat')
      e.wet_gas = 1;
      rv = fluid.rvSat(p);
      e.rvSat = rv;
      e.bG   = fluid.bG(p, rv, true(size(p)));
      e.muG  = fluid.muG(p, rv, true(size(p)));
      e.bG_usat  = fluid.bG(p, rv/2, false(size(p)));
      e.muG_usat = fluid.muG(p, rv/2, false(size(p)));
   else
      e.wet_gas = 0;
      e.bG  = fluid.bG(p);
      e.muG = fluid.muG(p);
   end

   % Saturation functions, probed across and beyond [0, 1].
   s = reshape(linspace(-0.1, 1.1, 25), [], 1);
   e.s = s;
   e.krW  = fluid.krW(s);
   e.krG  = fluid.krG(s);
   e.krOW = fluid.krOW(s);
   e.krOG = fluid.krOG(s);
   if isfield(fluid, 'pcOW'), e.pcOW = fluid.pcOW(s); end
   if isfield(fluid, 'pcOG'), e.pcOG = fluid.pcOG(s); end

   % End points, which the scaling machinery reads.
   if isfield(fluid, 'krPts')
      for f = reshape(fieldnames(fluid.krPts), 1, [])
         e.(['krPts_', f{1}]) = fluid.krPts.(f{1});
      end
   end

   out.(name) = e;
end

save(fullfile(thisDir, 'ref_ad_props_deck_fluid.mat'), '-struct', 'out', '-v7');
fprintf('Wrote ref_ad_props_deck_fluid.mat for %d decks\n', numel(decks));
