classdef ValidationCNVModel
    properties
        operators
        toleranceMB = 0.25
        toleranceCNV = 0.5
        FlowPropertyFunctions
    end
    methods
        function model = ValidationCNVModel()
            model.operators = struct('pv', [10; 20]);
            model.FlowPropertyFunctions = struct( ...
                'getProperty', @(model, state, name) ...
                    {[2; 4], [0.5; 0.5]});
        end
        function [phases, names] = getPhaseNames(model)
            phases = ['W', 'O'];
            names = {'water', 'oil'};
        end
    end
end
