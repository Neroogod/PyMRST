classdef CaptureAMGCL_CPRSolverAD < AMGCL_CPRSolverAD
    % AMGCL_CPRSolverAD with the AMGCL call replaced by a direct solve, so
    % the system MRST prepares for AMGCL can be recorded where AMGCL's MEX
    % cannot be built (validate_ad_oo_blackoil_tutorial_mex_acceleration.py).
    properties
        A_captured
        b_captured
        weights_captured
    end
    methods
        function solver = CaptureAMGCL_CPRSolverAD(varargin)
            solver = solver@AMGCL_CPRSolverAD(varargin{:});
        end
        function [result, report] = solveLinearSystem(solver, A, b, varargin)
            solver.A_captured = A;
            solver.b_captured = b;
            solver.weights_captured = solver.amgcl_setup.drs_row_weights;
            result = A\b;
            report = solver.getSolveReport('Converged', true, 'Residual', 0, 'Iterations', 1);
        end
    end
end
