// GENERATED from PDB 2OKK by buildsite.py -- do not hand-edit.
// Coordinates in Angstrom, origin at the substrate Ca. Ring/imine/phosphate-free PLP core and
// the substrate are crystal-derived; the carboxyl C and its two O are placed at run time from R.
var SITE_FIXED_A = [
    ['N',  -0.5760,   6.1240,   0.8230],   // N1
    ['C',   0.1140,   5.2330,   1.6220],   // C2
    ['C',   0.4360,   3.9680,   1.1360],   // C3
    ['C',   0.0500,   3.6070,  -0.1560],   // C4
    ['C',  -0.6460,   4.5290,  -0.9430],   // C5
    ['C',  -0.9670,   5.7880,  -0.4460],   // C6
    ['O',   1.0460,   3.1690,   1.8570],   // O3
    ['C',   0.5100,   5.6480,   3.0210],   // C2'
    ['C',   0.3660,   2.2360,  -0.7020],   // C4'
    ['H',  -0.8068,   7.0505,   1.1817],   // N1-H  (pyridinium, THE SINK)
    ['H',  -1.5109,   6.4954,  -1.0544],   // C6-H
    ['H',   1.5969,   5.6713,   3.0997],   // C2'-H
    ['H',   0.1095,   6.6392,   3.2336],   // C2'-H
    ['H',   0.1095,   4.9330,   3.7397],   // C2'-H
    ['H',   0.8314,   2.4444,   1.2330],   // O3-H
    ['H',  -0.5780,   1.8485,  -0.3189],   // C4'-H
    ['N',   0.5130,   1.3690,   0.3070],   // N7  (imine N)
    ['C',   0.0000,   0.0000,   0.0000],   // Ca
    ['H',   0.9440,   0.3875,  -0.3831],   // Ca-H
    ['C',   0.2452,  -1.3211,  -0.7318],   // Cb
    ['H',   0.3933,  -2.1184,  -0.0035],   // Cb-H
    ['H',  -0.6162,  -1.5560,  -1.3570],   // Cb-H
    ['H',   1.1336,  -1.2312,  -1.3570],   // Cb-H
  ];
var SITE_ASP_B = [
    ['C',  -2.6860,   7.6150,   2.7390],   // ASP364 CG
    ['O',  -3.1950,   6.4700,   2.7070],   // ASP364 OD1
    ['O',  -1.7340,   7.9530,   1.9860],   // ASP364 OD2 (accepts from N1-H)
    ['C',  -3.2480,   8.6260,   3.7410],   // ASP364 CB (capped)
    ['H',  -3.0548,   8.2784,   4.7558],   // ASP364 CB-H
    ['H',  -2.7673,   9.5927,   3.5909],   // ASP364 CB-H
    ['H',  -4.3228,   8.7280,   3.5909],   // ASP364 CB-H
    ['H',  -3.5930,   5.5748,   2.6820],   // ASP364 OD1-H (points away from N1)
  ];
var SITE_UP = [-0.866070,-0.355494,0.351493];   // Dunathan axis: the carboxylate leaves along +up
var SITE_E1 = [-0.160288,0.863435,0.478318];   // in-plane reference for the two carboxyl O
var SITE_CENTRE = [-0.791031,3.873622,1.224103];  // centroid of the FULL site (A+Asp), in the Ca-origin frame.
                                     // Both models must use this same centre, or A and B sit on different grids.
var SITE_BOX = 34.6153;                 // a.u., fixed: sized once for model B at R_MAX=3.0
var SITE_EXTENT = 9.59;
// indices into the built atom list (the fixed frame comes first, then C(=O)O and its two O)
var SITE_IDX = {"CA": 17, "C4P": 8, "N1": 0, "N1H": 9, "N7": 16};
