% Reference for validate_convert_input_units.py
out = struct();
mk = @() struct('cartDims', [2 2 2], 'COORD', ones(54,1), 'ZCORN', ones(64,1), ...
                'PERMX', 100*ones(8,1), 'PORO', 0.2*ones(8,1));
names = {'metric','field','lab'};
for k = 1:numel(names)
   u = getUnitSystem(upper(names{k}));
   g = convertInputUnits(mk(), u);
   r = struct('COORD', g.COORD(1), 'ZCORN', g.ZCORN(1), ...
              'PERMX', g.PERMX(1), 'PORO', g.PORO(1));
   out.(names{k}) = r;
end
% GRIDUNIT overrides the system for lengths.
g = mk(); g.GRIDUNIT = {'METRES'};
h = convertInputUnits(g, getUnitSystem('FIELD'));
out.gridunit = struct('COORD', h.COORD(1), 'PERMX', h.PERMX(1));
% An unknown keyword must error.
g = mk(); g.WIBBLE = ones(8,1);
try
   convertInputUnits(g, getUnitSystem('METRIC'));
   out.unknown_ran = 1;
catch err
   out.unknown_ran = 0; out.unknown_msg = err.message;
end
save('/Users/Shared/PyMRST_workspace/PyMRST/scripts/matlab/ref_convert_units.mat', '-struct', 'out', '-v7');
fprintf('wrote ref_convert_units.mat\n');
