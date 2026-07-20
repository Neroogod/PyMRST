classdef Phase119Component
    properties
        phaseFractions
        pureDensities
    end
    methods
        function component = Phase119Component(phaseFractions, pureDensities)
            component.phaseFractions = phaseFractions;
            component.pureDensities = pureDensities;
        end
        function frac = getPhaseCompositionSurface(component, model, state, pressure, T)
            frac = cell(1, numel(component.phaseFractions));
            for i = 1:numel(component.phaseFractions)
                frac{i} = component.phaseFractions(i);
            end
        end
        function rho = getPurePhaseDensitySurface(component, model, state, pressure, T)
            rho = cell(1, numel(component.pureDensities));
            for i = 1:numel(component.pureDensities)
                rho{i} = component.pureDensities(i);
            end
        end
    end
end
