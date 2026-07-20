function dump_phase119_reference()
% Phase 119 reference: BasicSeparator mass-mode split.

pymrstRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
mrstRoot = fullfile(fileparts(pymrstRoot), 'mrst-2024a-Geomech');
run(fullfile(mrstRoot, 'startup.m'));
mrstModule add ad-core

pressure = 1*atm;
temperature = 300;
component_stream = [2.0, 3.0];

components = { ...
    Phase119Component([0.20, 0.80], [700.0, 5.0]), ...
    Phase119Component([0.90, 0.10], [850.0, 15.0]) ...
};
model = Phase119Model(components, 2);
sep = BasicSeparator('pressure', pressure, 'T', temperature);

mass_stream_cell = localVectorToCell(component_stream);
[phase_mass_streams_cell, phase_mass_fractions_cell, densities_cell] = ...
    sep.separateComponentMassStream(model, mass_stream_cell);

phase_mass_streams = localCellScalars(phase_mass_streams_cell);
phase_mass_fractions = localPhaseComponentMatrix(phase_mass_fractions_cell);
densities = localCellScalars(densities_cell);
surface_rates = phase_mass_streams ./ densities;

out = fullfile(fileparts(mfilename('fullpath')), 'ref_p119_basic_separator_mass.mat');
save(out, 'pressure', 'temperature', 'component_stream', ...
    'phase_mass_streams', 'phase_mass_fractions', 'densities', 'surface_rates');
fprintf('Wrote ref_p119_basic_separator_mass.mat to %s\n', fileparts(out));
end

function c = localVectorToCell(v)
    c = cell(1, numel(v));
    for i = 1:numel(v)
        c{i} = v(i);
    end
end

function x = localCellScalars(c)
    x = zeros(1, numel(c));
    for i = 1:numel(c)
        x(i) = c{i};
    end
end

function x = localPhaseComponentMatrix(c)
    nph = numel(c);
    ncomp = numel(c{1});
    x = zeros(nph, ncomp);
    for ph = 1:nph
        for comp = 1:ncomp
            x(ph, comp) = c{ph}{comp};
        end
    end
end
