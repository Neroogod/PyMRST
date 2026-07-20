classdef Phase119Model
    properties
        Components
        nph
    end
    methods
        function model = Phase119Model(components, nph)
            model.Components = components;
            model.nph = nph;
        end
        function n = getNumberOfPhases(model)
            n = model.nph;
        end
    end
end
