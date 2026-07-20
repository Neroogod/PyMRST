% Phase 18 reference: CO2-brine property correlations (MRST co2lab-mit)
%
% Redlich-Kwong CO2 density, Fenghour CO2 viscosity, Rowe-Chou brine density,
% Islam-Carlson brine+CO2 viscosity, Duan-Sun activity, Davidson gas mixture.
% All scalar in MATLAB, so evaluated point by point over a grid.

mrstModule add co2lab-mit ad-core
here = fileparts(mfilename('fullpath'));

T = [288.15, 303.15, 313.15, 333.15, 353.15, 368.15]';   % K (spans the 313.15 switch)
P = [1, 20, 60, 100, 200, 400]';                         % bar
S = [0, 0.05, 0.15, 0.25]';                              % NaCl mass fraction
M = [0, 0.5, 1.0, 2.0, 4.0]';                            % NaCl molality
W = [0, 0.01, 0.03]';                                    % dissolved CO2 mass fraction

% --- CO2 density and viscosity over (T, P) -------------------------------
[TT, PP] = ndgrid(T, P);
TT = TT(:); PP = PP(:);
rho_co2 = zeros(size(TT)); mu_co2 = zeros(size(TT));
for k = 1:numel(TT)
    [~, ~, rho_co2(k)] = pvtCO2RK1949(TT(k), PP(k));
    mu_co2(k) = viscCO2F1998(TT(k), rho_co2(k));
end

% --- brine density over (T, P, S) ----------------------------------------
[T2, P2, S2] = ndgrid(T, P, S);
T2 = T2(:); P2 = P2(:); S2 = S2(:);
rho_b = zeros(size(T2)); c_b = zeros(size(T2));
for k = 1:numel(T2)
    [rho_b(k), c_b(k)] = pvtBrineRoweChou1970(T2(k), P2(k), S2(k));
end

% --- brine + CO2 viscosity over (T, P, m, w) ------------------------------
[T3, P3, M3, W3] = ndgrid(T, P, M, W);
T3 = T3(:); P3 = P3(:); M3 = M3(:); W3 = W3(:);
mu_bc = zeros(size(T3));
for k = 1:numel(T3)
    mu_bc(k) = viscBrineCO2IC2012(T3(k), P3(k), M3(k), W3(k));
end

% --- activity coefficient over (T, P, molality) ---------------------------
[T4, P4, M4] = ndgrid(T, P, M);
T4 = T4(:); P4 = P4(:); M4 = M4(:);
gam = zeros(size(T4));
for k = 1:numel(T4)
    m_io = [M4(k), 0, 0, 0, M4(k), 0];      % pure NaCl brine
    gam(k) = activityCO2DS2003(T4(k), P4(k), m_io);
end

% --- gas mixture viscosity -----------------------------------------------
xs  = [1 0; 0.7 0.3; 0.5 0.5; 0.2 0.8];
Mw  = [44.0095e-3, 18.015e-3];
mus = [1.5e-5, 1.0e-5];
mu_mix = zeros(size(xs,1),1);
for k = 1:size(xs,1)
    mu_mix(k) = viscGasMixtD1993(xs(k,:), Mw, mus);
end

ref = struct('T', TT, 'P', PP, 'rho_co2', rho_co2, 'mu_co2', mu_co2, ...
             'T2', T2, 'P2', P2, 'S2', S2, 'rho_b', rho_b, 'c_b', c_b, ...
             'T3', T3, 'P3', P3, 'M3', M3, 'W3', W3, 'mu_bc', mu_bc, ...
             'T4', T4, 'P4', P4, 'M4', M4, 'gam', gam, ...
             'xs', xs, 'Mw', Mw, 'mus', mus, 'mu_mix', mu_mix);
save(fullfile(here, 'ref_p18_co2_brine_props.mat'), '-struct', 'ref', '-v7');

fprintf(['Wrote Phase-18 reference: CO2 rho %.1f-%.1f kg/m3, mu %.1f-%.1f uPa.s;' ...
         ' brine rho %.0f-%.0f kg/m3\n'], min(rho_co2), max(rho_co2), ...
        min(mu_co2)*1e6, max(mu_co2)*1e6, min(rho_b), max(rho_b));
