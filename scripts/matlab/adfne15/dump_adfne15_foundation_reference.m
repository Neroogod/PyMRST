function dump_adfne15_foundation_reference(output_path, module_path)
% Direct deterministic ADFNE 1.5 geometry reference for the PyMRST port.
addpath(module_path);
global Tolerance Silent Report
Tolerance = 1e-14; Silent = true; Report = false;

angles360 = Angle([-10; 0; 90; 180; 370], 360);
angles90 = Angle([-10; 45; 100; 190], 90);
scaled = Scale([0; .5; 1], 0, 1, 10, 20);
lines = [-1,.5,2,.5; .2,.2,.8,.8; -2,-2,-1,-1];
[clipped, clip_mask] = Clip(lines, [0,0,1,1]);
line_lengths = Length(clipped);
line_centers = Center(clipped);
line_bbox = Bbox(clipped, 'pp');
line_orientations = [Orientation(clipped).Angle]';
points = [0,0; 1,0; 1,1];
poly_lines = Polyline(points);
point_line = Pointline(points, uint32([1,3]));
clusters = Cluster([1,2;4,5;2,3]);
network = [0,.5,1,.5; .25,0,.25,1; .75,0,.75,1; .1,.1,.2,.1];
[network_xts, network_ids, network_labels] = Intersect(network);
snap_points = [.2,.01; .8,.9];
snap_lines = [0,0,1,0; 0,1,1,1];
snap_distances = Snap(snap_points, snap_lines, 0);
[snapped_points, snap_mask] = Snap(snap_points, snap_lines, .11);
isolated_mask = Isolated([0,0,1,0; 1,0,2,0; 2,0,0,0]);
flow_be = [0,0,0,0,1,0; 1,0,0,1,1,0];
boundary_values = Boundary([0,.25,0; .5,.5,0; 1,.75,0], flow_be, [1,0]);
g.Node(1).Edges = 1; g.Node(1).Nodes = 2; g.Node(1).Location = [0,.5,0]; g.Node(1).Type = 1; g.Node(1).Pressure = NaN;
g.Node(2).Edges = [1,2]; g.Node(2).Nodes = [1,3]; g.Node(2).Location = [.5,.5,0]; g.Node(2).Type = 0; g.Node(2).Pressure = NaN;
g.Node(3).Edges = 2; g.Node(3).Nodes = 2; g.Node(3).Location = [1,.5,0]; g.Node(3).Type = 2; g.Node(3).Pressure = NaN;
for ii=1:2
    g.Edge(ii).Nodes=[ii,ii+1]; g.Edge(ii).Type=0; g.Edge(ii).Depth=1;
    g.Edge(ii).Length=.5; g.Edge(ii).Aperture=1e-3; g.Edge(ii).Dip=0; g.Edge(ii).Dir=0;
end
g.Size=[3,2]; g.BE=flow_be; flow_dfn.grh=g;
flow_dfn = Solve(flow_dfn, [1,0], 1.787e-6);
flow_pressures = [flow_dfn.grh.Node.Pressure]';
flow_drops = [flow_dfn.grh.Edge.dP]';
flow_rates = [flow_dfn.grh.Edge.Flow]';
flow_permeability = [flow_dfn.grh.Edge.Permeability]';
flow_force = [flow_dfn.grh.Edge.Force]';
left_poly = [0,0,0;0,.4,0;0,.4,1;0,0,1]; right_poly=left_poly; right_poly(:,1)=1;
middle_poly = [0,.1,.5;1,.1,.5;1,.8,.5;0,.8,.5];
pipe3 = Pipe({left_poly;right_poly},{middle_poly},'cnt');
pipe3_lines=pipe3.pip.Pipe; pipe3_types=pipe3.pip.Type;
pipe3_depths=pipe3.pip.Depth; pipe3_pairs=pipe3.dfn.xID;
pipe3_intersections=pipe3.dfn.xLine; pipe3_labels=pipe3.dfn.Label;
net2=Net(1:3,4:5); net3=Net(1:2,3:4,5:6);
stacked=Stack({[1,2];[3,4]}); reshaped=Reshape(1:16,4,4);
flattened_triangle=Flatri(reshaped); filtered_ids=Filter([1,2;2,3;2,3]);
not_values=Not([1,0,4,3,2,1],[1,3]); [naned_x,naned_y]=Naned([0,1,2,3]);
mesh_unavailable=false; try Mesh([0,0,1,1],.1,false,'r'); catch; mesh_unavailable=true; end
translated=Translate([0,0;1,0;1,1],[2,3]);
rotated=Rotate([0,0;1,0;1,1],'ang',90,'cnt',[0,0]);
rays=Ray([0,0;1,0;1,1]); distances=Distance([.5,.5;2,.5],{[0,0;1,0;1,1]});
labels=Label({[1,3],[4]},5); relabeled=Relabel([1,1,-1,2,2,-2,1,2,3]);
wrapped=Wrap(1:10,4); tensor_unavailable=false;
try Tensor(struct()); catch; tensor_unavailable=true; end
inpoly2=Inpoly([.5,.5;1,.5;2,.5],[0,0;1,0;1,1;0,1],1e-14);
inpoly3=Inpoly([.5,.5,0;.5,.5,.1],[0,0,0;1,0,0;1,1,0;0,1,0],1e-8);
ellipse4=Geometry('ellipse','cnt',[1,2],'r',[2,1],'q',4);
sphere6=Geometry('sphere','cnt',[1,2,3],'r',2,'q',6);
cylinder=GetShapePoly('shape','Cylinder','Radius',1,'Center',[0,0,0], ...
    'Length',1,'Resolution',4);
[clipped_domain,clipped_domain_mask]=PolyClip({[.5,-2,-2;.5,2,-2;.5,2,2;.5,-2,2]},cylinder);
clipped_domain_polygon=clipped_domain{1}; cylinder_cap1=cylinder{1}; cylinder_side1=cylinder{3};
convert_clock=Convert('clock',1000); convert_deg=Convert('deg',pi);
convert_rad=Convert('rad',180); convert_norm=Convert('norm',[3,4]);
convert_tri_all=Convert('tri',[0,0;1,0;1,1;0,1]); convert_tri1=convert_tri_all{1}{1};
convert_3d=Convert('3d',[1,2;3,4]);
convert_2d=Convert('2d',[0,0,0;1,0,0;1,1,0;0,1,0]);
option_result=Option({'fc','red','r',2},'fc','blue','lw',1);
option_radius=option_result.radius; option_linewidth=option_result.linewidth;
cdata_almost=CData.Almost([0,1],0,.1); [cdata_x,cdata_y]=CData.Deal([1,2;3,4]);
cdata_get=CData.Get([1,2;3,4],10); [cdata_less,cdata_equal,cdata_greater]=CData.Leg([.4,.5,.6],.5,.01);
cdata_map=CData.Map([1,2,3],@max,@min,@mean); cdata_array_all=CData.Array([0,0,1,0],[1,2;3,4]);
cdata_array1=cdata_array_all{1};
geo_lines=[0,0,1,0;0,0,0,2]; geo_info=Info(geo_lines);
stats_toolbox_available=true;
try
    [stats_min,stats_mean,stats_max,stats_std,stats_count]=Stats([1,2,NaN,4],'fixture');
catch
    stats_toolbox_available=false; stats_min=NaN; stats_mean=NaN;
    stats_max=NaN; stats_std=NaN; stats_count=NaN;
end
geo_points=[0,0;1,0;0,1;1,1]; geo_values=[1;2;3;4];
[cloud_d,cloud_g,cloud_v]=Variocloud(geo_points,geo_values,[],false);
[variogram_h,variogram_v,variogram_p]=Variogram(cloud_d,cloud_g,cloud_v,.5,false);
model_lags=[0,.5,1,2]; model_sph=Variomodel({'name','sph','nugget',0,'sill',2,'range',1},model_lags,false);
model_exp=Variomodel({'name','exp','nugget',.1,'sill',2,'range',3},model_lags,false);
model_lin=Variomodel({'name','lin','nugget',1,'slope',2},model_lags,false);
markov_fixed=Markov([1,0;0,1],1,10);
upscale_arithmetic=Upscaling(reshape(1:16,4,4)','arithmetic',2);
kriging_toolbox_available=true;
try Kriging(geo_points,geo_values,cloud_d,{'name','sph','nugget',0,'sill',2,'range',2},[2,2],false); catch; kriging_toolbox_available=false; end
grid4=Grid(4);
cext_plane=[0,0,0,1,0,0,0,1,0]; cext_normal=CExt.planeNormal(cext_plane);
cext_line_plane=CExt.LineXPlane([0,0,1,0,0,-2],cext_plane);
cext_projected=CExt.PointOnPlane([1,2,3],cext_plane);

save(output_path, 'angles360', 'angles90', 'scaled', 'clipped', 'clip_mask', ...
    'line_lengths', 'line_centers', 'line_bbox', 'line_orientations', ...
    'poly_lines', 'point_line', 'clusters', 'network_xts', 'network_ids', ...
    'network_labels', ...
    'snap_distances', 'snapped_points', 'snap_mask', 'isolated_mask', ...
    'boundary_values', 'flow_pressures', 'flow_drops', 'flow_rates', ...
    'flow_permeability', 'flow_force', 'pipe3_lines', 'pipe3_types', ...
    'pipe3_depths', 'pipe3_pairs', 'pipe3_intersections', 'pipe3_labels', ...
    'net2', 'net3', 'stacked', 'reshaped', 'flattened_triangle', ...
    'filtered_ids', 'not_values', 'naned_x', 'naned_y', 'mesh_unavailable', ...
    'translated', 'rotated', 'rays', 'distances', 'labels', 'relabeled', ...
    'wrapped', 'tensor_unavailable', 'inpoly2', 'inpoly3', 'ellipse4', ...
    'sphere6', 'clipped_domain_polygon', 'clipped_domain_mask', ...
    'cylinder_cap1', 'cylinder_side1', 'convert_clock', 'convert_deg', ...
    'convert_rad', 'convert_norm', 'convert_tri1', 'convert_3d', ...
    'convert_2d', 'option_radius', 'option_linewidth', 'cdata_almost', ...
    'cdata_x', 'cdata_y', 'cdata_get', 'cdata_less', 'cdata_equal', ...
    'cdata_greater', 'cdata_map', 'cdata_array1', 'geo_info', ...
    'stats_min', 'stats_mean', 'stats_max', 'stats_std', 'stats_count', ...
    'stats_toolbox_available', ...
    'cloud_d', 'cloud_g', 'cloud_v', 'variogram_h', 'variogram_v', ...
    'variogram_p', 'model_lags', 'model_sph', 'model_exp', 'model_lin', ...
    'markov_fixed', 'upscale_arithmetic', 'kriging_toolbox_available', ...
    'grid4', 'cext_normal', 'cext_line_plane', 'cext_projected', '-v7');
end
