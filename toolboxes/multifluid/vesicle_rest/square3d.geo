h = 0.16;
xmin = -2.;
xmax = 2.;
ymin = -2.;
ymax = 2.;
zmin = -2.;
zmax = 2.;

Point(1) = {xmin,ymin,zmin,h};
Point(2) = {xmax,ymin,zmin,h};
Point(3) = {xmax,ymax,zmin,h};
Point(4) = {xmin,ymax,zmin,h};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line Loop(1) = {1:4};

Plane Surface(1) = {1};

out[] = Extrude{0,0,zmax-zmin}{ Surface{1}; };

Surface Loop(301) = {201, 202, 203, 204, 205, 206};

Field[1] = Box;
Field[1].VIn = h/4;
Field[1].VOut = h;
Field[1].XMin = -0.8;
Field[1].XMax = 0.8;
Field[1].YMin = -0.8;
Field[1].YMax = 0.8;
Field[1].ZMin = -0.8;
Field[1].ZMax = 0.8;
Background Field = 1;

Physical Surface("Left") = {13};
Physical Surface("Bottom") = {1};
Physical Surface("Right") = {21};
Physical Surface("Top") = {26};
Physical Surface("Front") = {17};
Physical Surface("Back") = {25};

Physical Volume("Omega") = {1};
