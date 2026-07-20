classdef Phase120ControlModel
    properties
        water
        oil
        gas
    end
    methods
        function self = Phase120ControlModel(water, oil, gas)
            self.water = water;
            self.oil = oil;
            self.gas = gas;
        end
        function phases = getActivePhases(self)
            phases = [self.water, self.oil, self.gas];
        end
    end
end
