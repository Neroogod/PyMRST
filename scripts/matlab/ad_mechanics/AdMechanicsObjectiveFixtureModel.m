classdef AdMechanicsObjectiveFixtureModel
    properties
        G
        mechModel
        fluidModel
    end
    methods
        function model = AdMechanicsObjectiveFixtureModel()
            model.G.griddim = 2;
            model.G.nodes.num = 3;
            model.G.cartDims = [1, 1];
            model.mechModel.operators.isdirdofs = logical([1; 1; 1; 0; 0; 0]);
            model.fluidModel.FacilityModel = AdMechanicsObjectiveFixtureFacility();
        end
        function varargout = getProps(~, state, varargin)
            for k = 1:numel(varargin)
                name = varargin{k};
                if strcmp(name, 'water'), name = 'sW'; end
                varargout{k} = state.(name);
            end
        end
    end
end
