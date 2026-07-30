classdef ValidationCNVProblem
    properties
        state = struct()
        dt = 2
        equations
    end
    methods
        function problem = ValidationCNVProblem()
            problem.equations = {[99], [1; -3], [2; 2]};
        end
        function count = numeq(problem)
            count = numel(problem.equations);
        end
        function index = indexOfEquationName(problem, name)
            switch name
                case 'water'
                    index = 2;
                case 'oil'
                    index = 3;
                otherwise
                    error('Unknown equation');
            end
        end
    end
end
