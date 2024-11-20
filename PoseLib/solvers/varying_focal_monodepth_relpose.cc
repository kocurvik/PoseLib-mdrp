//
// Created by kocur on 19-Nov-24.
//

#include "varying_focal_monodepth_relpose.h"

#include "PoseLib/misc/decompositions.h"
#include "PoseLib/misc/essential.h"
#include "relpose_7pt.h"

#include <iostream>
namespace poselib {

void varying_focal_fundamental_relpose(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                                       std::vector<ImagePair> *models) {

    std::vector<Eigen::Vector3d> x1h(7);
    std::vector<Eigen::Vector3d> x2h(7);
    for (int i = 0; i < 7; ++i) {
        x1h[i] = x1[i].homogeneous().normalized();
        x2h[i] = x2[i].homogeneous().normalized();
    }

    std::vector<Eigen::Matrix3d> Fs;
    relpose_7pt(x1h, x2h, &Fs);

    models->reserve(Fs.size());

    for (const Eigen::Matrix3d& F : Fs) {
        std::pair<Camera, Camera> cameras =
            focals_from_fundamental(F, Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero());
        Camera camera1 = cameras.first;
        Camera camera2 = cameras.second;

        if (std::isnan(camera1.focal()))
            return;
        if (std::isnan(camera2.focal()))
            return;

        Eigen::DiagonalMatrix<double, 3> K1(camera1.focal(), camera1.focal(), 1.0);
        Eigen::DiagonalMatrix<double, 3> K2(camera2.focal(), camera2.focal(), 1.0);

        Eigen::Matrix3d E = K2 * F * K1;

        std::vector<CameraPose> poses;
        motion_from_essential(E, x1h, x2h, &poses);

        for (const CameraPose &pose : poses) {
            models->emplace_back(pose, camera1, camera2);
        }
    }
}

void varying_focal_monodepth_relpose(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                                     const std::vector<Eigen::Vector2d> &sigma,
                                     std::vector<ImagePair> *models) {
    models->clear();
    std::vector<Eigen::Vector3d> x1h(4);
    std::vector<Eigen::Vector3d> x2h(4);
    for (int i = 0; i < 4; ++i) {
        x1h[i] = x1[i].homogeneous().normalized();
        x2h[i] = x2[i].homogeneous().normalized();
    }

    Eigen::MatrixXd coefficients(12, 12);
    int i;

    // Form a linear system: i-th row of A(=a) represents
    // the equation: (m2[i], 1)'*F*(m1[i], 1) = 0
    size_t row = 0;
    for (i = 0; i < 4; i++)
    {
        double u11 = x1[i](0), v11 = x1[i](1), u12 = x2[i](0), v12 = x2[i](1);
        double q1 = sigma[i](0), q2 = sigma[i](1);
        double q = q1 / q2;

        coefficients(row, 0) = -u11;
        coefficients(row, 1) = -v11;
        coefficients(row, 2) = -1;
        coefficients(row, 3) = 0;
        coefficients(row, 4) = 0;
        coefficients(row, 5) = 0;
        coefficients(row, 6) = 0;
        coefficients(row, 7) = 0;
        coefficients(row, 8) = 0;
        coefficients(row, 9) = 0;
        coefficients(row, 10) = q;
        coefficients(row, 11) = -q * v12;
        ++row;

        coefficients(row, 0) = 0;
        coefficients(row, 1) = 0;
        coefficients(row, 2) = 0;
        coefficients(row, 3) = -u11;
        coefficients(row, 4) = -v11;
        coefficients(row, 5) = -1;
        coefficients(row, 6) = 0;
        coefficients(row, 7) = 0;
        coefficients(row, 8) = 0;
        coefficients(row, 9) = -q;
        coefficients(row, 10) = 0;
        coefficients(row, 11) = q * u12;
        ++row;

        if (i == 3)
            break;

        coefficients(row, 0) = 0;
        coefficients(row, 1) = 0;
        coefficients(row, 2) = 0;
        coefficients(row, 3) = 0;
        coefficients(row, 4) = 0;
        coefficients(row, 5) = 0;
        coefficients(row, 6) = -u11;
        coefficients(row, 7) = -v11;
        coefficients(row, 8) = -1;
        coefficients(row, 9) = q * v12;
        coefficients(row, 10) = -q * u12;
        coefficients(row, 11) = 0;
        ++row;
    }

    Eigen::Matrix<double, 12, 1> f1 = coefficients.block<11, 11>(0, 0).partialPivLu().solve(-coefficients.block<11, 1>(0, 11)).homogeneous();

    Eigen::Matrix3d F;
    F << f1[0], f1[1], f1[2], f1[3], f1[4], f1[5], f1[6], f1[7], f1[8];

//    std::cout << "F: " << std::endl << F << std::endl;
//    std::cout << "Ep: " << x2h[0].transpose() * F * x1h[0] << std::endl;
//    std::cout << "Det: " << F.determinant() << std::endl;

    std::pair<Camera, Camera> cameras = focals_from_fundamental(F, Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero());
    Camera camera1 = cameras.first;
    Camera camera2 = cameras.second;

    if (std::isnan(camera1.focal()))
        return;
    if (std::isnan(camera2.focal()))
        return;

    Eigen::DiagonalMatrix<double, 3> K1(camera1.focal(), camera1.focal(), 1.0);
    Eigen::DiagonalMatrix<double, 3> K2(camera2.focal(), camera2.focal(), 1.0);

    Eigen::Matrix3d E = K2 * F * K1;

    std::vector<CameraPose> poses;
    motion_from_essential(E, x1h, x2h, &poses);

    models->reserve(poses.size());

    for (const CameraPose& pose : poses){
        models->emplace_back(pose, camera1, camera2);
    }
}

Eigen::MatrixXd varying_focal_monodepth_gj(Eigen::VectorXd d){
    Eigen::VectorXd coeffs(40);
    coeffs[0] = std::pow(d[8],2) - 2*d[8]*d[9] + std::pow(d[9],2) + std::pow(d[12],2) - 2*d[12]*d[13] + std::pow(d[13],2);
    coeffs[1] = -std::pow(d[0],2) + 2*d[0]*d[1] - std::pow(d[1],2) - std::pow(d[4],2) + 2*d[4]*d[5] - std::pow(d[5],2);
    coeffs[2] = 2*std::pow(d[8],2)*d[20] - 2*d[8]*d[9]*d[20] + 2*std::pow(d[12],2)*d[20] - 2*d[12]*d[13]*d[20] - 2*d[8]*d[9]*d[21] + 2*std::pow(d[9],2)*d[21] - 2*d[12]*d[13]*d[21] + 2*std::pow(d[13],2)*d[21];
    coeffs[3] = -2*std::pow(d[0],2)*d[16] + 2*d[0]*d[1]*d[16] - 2*std::pow(d[4],2)*d[16] + 2*d[4]*d[5]*d[16] + 2*d[0]*d[1]*d[17] - 2*std::pow(d[1],2)*d[17] + 2*d[4]*d[5]*d[17] - 2*std::pow(d[5],2)*d[17];
    coeffs[4] = std::pow(d[8],2)*std::pow(d[20],2) + std::pow(d[12],2)*std::pow(d[20],2) - 2*d[8]*d[9]*d[20]*d[21] - 2*d[12]*d[13]*d[20]*d[21] + std::pow(d[9],2)*std::pow(d[21],2) + std::pow(d[13],2)*std::pow(d[21],2);
    coeffs[5] = std::pow(d[20],2) - 2*d[20]*d[21] + std::pow(d[21],2);
    coeffs[6] = -std::pow(d[0],2)*std::pow(d[16],2) - std::pow(d[4],2)*std::pow(d[16],2) + 2*d[0]*d[1]*d[16]*d[17] + 2*d[4]*d[5]*d[16]*d[17] - std::pow(d[1],2)*std::pow(d[17],2) - std::pow(d[5],2)*std::pow(d[17],2);
    coeffs[7] = -std::pow(d[16],2) + 2*d[16]*d[17] - std::pow(d[17],2);
    coeffs[8] = std::pow(d[8],2) - 2*d[8]*d[10] + std::pow(d[10],2) + std::pow(d[12],2) - 2*d[12]*d[14] + std::pow(d[14],2);
    coeffs[9] = -std::pow(d[0],2) + 2*d[0]*d[2] - std::pow(d[2],2) - std::pow(d[4],2) + 2*d[4]*d[6] - std::pow(d[6],2);
    coeffs[10] = 2*std::pow(d[8],2)*d[20] - 2*d[8]*d[10]*d[20] + 2*std::pow(d[12],2)*d[20] - 2*d[12]*d[14]*d[20] - 2*d[8]*d[10]*d[22] + 2*std::pow(d[10],2)*d[22] - 2*d[12]*d[14]*d[22] + 2*std::pow(d[14],2)*d[22];
    coeffs[11] = -2*std::pow(d[0],2)*d[16] + 2*d[0]*d[2]*d[16] - 2*std::pow(d[4],2)*d[16] + 2*d[4]*d[6]*d[16] + 2*d[0]*d[2]*d[18] - 2*std::pow(d[2],2)*d[18] + 2*d[4]*d[6]*d[18] - 2*std::pow(d[6],2)*d[18];
    coeffs[12] = std::pow(d[8],2)*std::pow(d[20],2) + std::pow(d[12],2)*std::pow(d[20],2) - 2*d[8]*d[10]*d[20]*d[22] - 2*d[12]*d[14]*d[20]*d[22] + std::pow(d[10],2)*std::pow(d[22],2) + std::pow(d[14],2)*std::pow(d[22],2);
    coeffs[13] = std::pow(d[20],2) - 2*d[20]*d[22] + std::pow(d[22],2);
    coeffs[14] = -std::pow(d[0],2)*std::pow(d[16],2) - std::pow(d[4],2)*std::pow(d[16],2) + 2*d[0]*d[2]*d[16]*d[18] + 2*d[4]*d[6]*d[16]*d[18] - std::pow(d[2],2)*std::pow(d[18],2) - std::pow(d[6],2)*std::pow(d[18],2);
    coeffs[15] = -std::pow(d[16],2) + 2*d[16]*d[18] - std::pow(d[18],2);
    coeffs[16] = std::pow(d[8],2) - 2*d[8]*d[11] + std::pow(d[11],2) + std::pow(d[12],2) - 2*d[12]*d[15] + std::pow(d[15],2);
    coeffs[17] = -std::pow(d[0],2) + 2*d[0]*d[3] - std::pow(d[3],2) - std::pow(d[4],2) + 2*d[4]*d[7] - std::pow(d[7],2);
    coeffs[18] = 2*std::pow(d[8],2)*d[20] - 2*d[8]*d[11]*d[20] + 2*std::pow(d[12],2)*d[20] - 2*d[12]*d[15]*d[20] - 2*d[8]*d[11]*d[23] + 2*std::pow(d[11],2)*d[23] - 2*d[12]*d[15]*d[23] + 2*std::pow(d[15],2)*d[23];
    coeffs[19] = -2*std::pow(d[0],2)*d[16] + 2*d[0]*d[3]*d[16] - 2*std::pow(d[4],2)*d[16] + 2*d[4]*d[7]*d[16] + 2*d[0]*d[3]*d[19] - 2*std::pow(d[3],2)*d[19] + 2*d[4]*d[7]*d[19] - 2*std::pow(d[7],2)*d[19];
    coeffs[20] = std::pow(d[8],2)*std::pow(d[20],2) + std::pow(d[12],2)*std::pow(d[20],2) - 2*d[8]*d[11]*d[20]*d[23] - 2*d[12]*d[15]*d[20]*d[23] + std::pow(d[11],2)*std::pow(d[23],2) + std::pow(d[15],2)*std::pow(d[23],2);
    coeffs[21] = std::pow(d[20],2) - 2*d[20]*d[23] + std::pow(d[23],2);
    coeffs[22] = -std::pow(d[0],2)*std::pow(d[16],2) - std::pow(d[4],2)*std::pow(d[16],2) + 2*d[0]*d[3]*d[16]*d[19] + 2*d[4]*d[7]*d[16]*d[19] - std::pow(d[3],2)*std::pow(d[19],2) - std::pow(d[7],2)*std::pow(d[19],2);
    coeffs[23] = -std::pow(d[16],2) + 2*d[16]*d[19] - std::pow(d[19],2);
    coeffs[24] = std::pow(d[9],2) - 2*d[9]*d[10] + std::pow(d[10],2) + std::pow(d[13],2) - 2*d[13]*d[14] + std::pow(d[14],2);
    coeffs[25] = -std::pow(d[1],2) + 2*d[1]*d[2] - std::pow(d[2],2) - std::pow(d[5],2) + 2*d[5]*d[6] - std::pow(d[6],2);
    coeffs[26] = 2*std::pow(d[9],2)*d[21] - 2*d[9]*d[10]*d[21] + 2*std::pow(d[13],2)*d[21] - 2*d[13]*d[14]*d[21] - 2*d[9]*d[10]*d[22] + 2*std::pow(d[10],2)*d[22] - 2*d[13]*d[14]*d[22] + 2*std::pow(d[14],2)*d[22];
    coeffs[27] = -2*std::pow(d[1],2)*d[17] + 2*d[1]*d[2]*d[17] - 2*std::pow(d[5],2)*d[17] + 2*d[5]*d[6]*d[17] + 2*d[1]*d[2]*d[18] - 2*std::pow(d[2],2)*d[18] + 2*d[5]*d[6]*d[18] - 2*std::pow(d[6],2)*d[18];
    coeffs[28] = std::pow(d[9],2)*std::pow(d[21],2) + std::pow(d[13],2)*std::pow(d[21],2) - 2*d[9]*d[10]*d[21]*d[22] - 2*d[13]*d[14]*d[21]*d[22] + std::pow(d[10],2)*std::pow(d[22],2) + std::pow(d[14],2)*std::pow(d[22],2);
    coeffs[29] = std::pow(d[21],2) - 2*d[21]*d[22] + std::pow(d[22],2);
    coeffs[30] = -std::pow(d[1],2)*std::pow(d[17],2) - std::pow(d[5],2)*std::pow(d[17],2) + 2*d[1]*d[2]*d[17]*d[18] + 2*d[5]*d[6]*d[17]*d[18] - std::pow(d[2],2)*std::pow(d[18],2) - std::pow(d[6],2)*std::pow(d[18],2);
    coeffs[31] = -std::pow(d[17],2) + 2*d[17]*d[18] - std::pow(d[18],2);
    coeffs[32] = std::pow(d[9],2) - 2*d[9]*d[11] + std::pow(d[11],2) + std::pow(d[13],2) - 2*d[13]*d[15] + std::pow(d[15],2);
    coeffs[33] = -std::pow(d[1],2) + 2*d[1]*d[3] - std::pow(d[3],2) - std::pow(d[5],2) + 2*d[5]*d[7] - std::pow(d[7],2);
    coeffs[34] = 2*std::pow(d[9],2)*d[21] - 2*d[9]*d[11]*d[21] + 2*std::pow(d[13],2)*d[21] - 2*d[13]*d[15]*d[21] - 2*d[9]*d[11]*d[23] + 2*std::pow(d[11],2)*d[23] - 2*d[13]*d[15]*d[23] + 2*std::pow(d[15],2)*d[23];
    coeffs[35] = -2*std::pow(d[1],2)*d[17] + 2*d[1]*d[3]*d[17] - 2*std::pow(d[5],2)*d[17] + 2*d[5]*d[7]*d[17] + 2*d[1]*d[3]*d[19] - 2*std::pow(d[3],2)*d[19] + 2*d[5]*d[7]*d[19] - 2*std::pow(d[7],2)*d[19];
    coeffs[36] = std::pow(d[9],2)*std::pow(d[21],2) + std::pow(d[13],2)*std::pow(d[21],2) - 2*d[9]*d[11]*d[21]*d[23] - 2*d[13]*d[15]*d[21]*d[23] + std::pow(d[11],2)*std::pow(d[23],2) + std::pow(d[15],2)*std::pow(d[23],2);
    coeffs[37] = std::pow(d[21],2) - 2*d[21]*d[23] + std::pow(d[23],2);
    coeffs[38] = -std::pow(d[1],2)*std::pow(d[17],2) - std::pow(d[5],2)*std::pow(d[17],2) + 2*d[1]*d[3]*d[17]*d[19] + 2*d[5]*d[7]*d[17]*d[19] - std::pow(d[3],2)*std::pow(d[19],2) - std::pow(d[7],2)*std::pow(d[19],2);
    coeffs[39] = -std::pow(d[17],2) + 2*d[17]*d[19] - std::pow(d[19],2);


    static const int coeffs_ind[] = {0,8,16,24,32,0,8,16,24,32,1,9,17,25,33,2,10,0,8,18,16,26,24,34,32,0,8,16,24,32,1,9,17,25,33,2,10,0,8,16,18,24,26,32,34,0,8,16,24,32,
                                     1,9,17,25,33,3,11,1,9,19,17,25,27,33,35,4,12,2,10,8,20,0,16,18,28,24,26,32,36,34,2,10,8,0,18,16,26,34,24,32,1,9,17,25,33,3,11,1,9,17,
                                     19,25,27,33,35,4,12,2,10,18,20,26,28,34,36,2,10,8,16,18,0,26,32,34,24,5,13,21,29,37,9,1,17,25,33,3,11,9,1,17,19,25,27,33,35,6,14,3,11,22,
                                     19,27,30,35,38,5,4,13,12,10,2,18,21,20,26,29,28,34,37,36,4,12,10,2,20,18,28,8,24,36,26,16,0,34,32,3,11,19,9,17,1,27,33,25,35,6,14,3,11,19,
                                     22,27,30,35,38,4,12,20,28,36,4,12,10,18,20,2,28,34,36,26,5,13,21,29,37,5,13,21,29,37,7,15,23,31,39,11,3,19,9,25,27,17,1,35,33,6,14,11,3,19,
                                     22,27,30,35,38,6,14,22,30,38,5,12,13,21,4,20,28,29,36,37,5,12,13,4,20,21,10,26,29,28,18,37,2,36,34,7,15,23,31,39,6,14,22,11,19,3,30,35,27,38,
                                     6,14,22,30,38,12,20,4,36,28,13,5,21,29,37,13,5,21,29,37,7,15,23,31,39,14,6,22,11,27,30,19,3,38,35,13,21,5,12,28,37,20,4,29,36,7,15,23,31,39,
                                     14,22,6,38,30,13,29,21,5,37,15,7,23,31,39,7,15,23,31,39,14,6,22,30,38,7,15,23,31,39,15,31,23,7,39,15,7,23,31,39,14,30,22,6,38,15,23,7,39,31};

    static const int C_ind[] = {0,2,16,28,47,51,55,74,86,98,100,102,116,128,147,150,152,153,158,166,176,178,188,197,199,204,209,219,227,231,251,255,274,286,298,301,305,306,312,314,324,334,336,346,348,357,363,373,383,390,
                                403,408,426,438,449,450,452,454,459,466,469,477,478,481,497,500,502,503,508,510,516,517,520,526,528,532,538,543,547,549,554,559,561,568,569,571,577,581,585,592,606,612,614,634,646,651,655,657,663,673,
                                674,683,686,690,698,701,705,706,712,714,724,734,736,746,748,757,763,765,772,773,775,783,787,790,794,800,802,816,828,847,860,867,870,882,893,903,908,911,918,921,926,935,938,942,949,950,952,954,959,966,
                                969,977,978,981,997,1001,1003,1005,1008,1010,1017,1020,1024,1026,1032,1036,1038,1043,1048,1049,1054,1059,1061,1068,1069,1071,1077,1079,1080,1081,1085,1089,1091,1092,1095,1106,1112,1114,1115,1122,1125,1134,1137,1144,1146,1151,1155,1157,1163,1173,
                                1174,1183,1186,1190,1198,1206,1212,1214,1234,1246,1257,1263,1265,1272,1273,1275,1283,1287,1290,1294,1303,1308,1326,1338,1349,1354,1359,1369,1377,1381,1400,1402,1416,1428,1447,1460,1467,1470,1479,1480,1482,1489,1491,1493,1495,1503,1508,1511,1518,1521,
                                1526,1535,1538,1542,1549,1554,1559,1569,1577,1581,1606,1610,1612,1614,1617,1620,1632,1634,1643,1646,1657,1661,1663,1668,1671,1673,1679,1680,1683,1685,1689,1690,1691,1692,1695,1701,1705,1724,1736,1748,1756,1762,1764,1765,1772,1775,1784,1787,1794,1796,
                                1807,1813,1823,1833,1840,1865,1872,1875,1887,1894,1910,1917,1920,1932,1943,1961,1968,1971,1985,1992,2003,2008,2026,2038,2049,2060,2067,2070,2079,2080,2082,2089,2091,2093,2095,2115,2122,2125,2129,2130,2137,2139,2141,2144,2145,2156,2162,2164,2184,2196,
                                2215,2222,2225,2237,2244,2279,2280,2289,2291,2295,2310,2317,2320,2332,2343,2354,2359,2369,2377,2381,2411,2418,2421,2435,2442,2457,2463,2473,2483,2490,2529,2530,2539,2541,2545,2561,2568,2571,2585,2592,2629,2630,2639,2641,2645,2665,2672,2675,2687,2694};

    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(50,54);
    for (int i = 0; i < 400; i++) {
        C(C_ind[i]) = coeffs(coeffs_ind[i]);
    }

    Eigen::MatrixXd C0 = C.leftCols(50);
    Eigen::MatrixXd C1 = C.rightCols(4);
    Eigen::MatrixXd C12 = C0.partialPivLu().solve(C1);
    Eigen::MatrixXd RR(7, 4);
    RR << -C12.bottomRows(3), Eigen::MatrixXd::Identity(4, 4);

    static const int AM_ind[] = {4,0,1,2};
    Eigen::MatrixXd AM(4, 4);
    for (int i = 0; i < 4; i++) {
        AM.row(i) = RR.row(AM_ind[i]);
    }

    Eigen::EigenSolver<Eigen::MatrixXd> es(AM);
    Eigen::ArrayXcd D = es.eigenvalues();
    Eigen::ArrayXXcd V = es.eigenvectors();
    V = (V / V.row(0).replicate(4, 1)).eval();

    Eigen::MatrixXd sols(5, 4);
    int m = 0;
    for (int k = 0; k < 4; ++k)
    {

        if (abs(D(k).imag()) > 0.001 || V(2, k).real() < 0.0 ||
            V(3, k).real() < 0.0 || abs(V(2, k).imag()) > 0.001 || abs(V(3, k).imag()) > 0.001)
            continue;

        sols(2, m) = D(k).real(); // v x3
        sols(3, m) = V(2, k).real(); // f x4
        sols(4, m) = V(3, k).real(); // m x5

        Eigen::MatrixXd A1(3, 3);
        A1 << coeffs[0]*sols(2, m)*sols(2, m)*sols(4, m) + coeffs[2]*sols(2, m)*sols(4, m) + coeffs[4]*sols(4, m) + coeffs[5], coeffs[1]*sols(3, m), coeffs[3]*sols(3, m),
            coeffs[8]*sols(2, m)*sols(2, m)*sols(4, m) + coeffs[10]*sols(2, m)*sols(4, m) + coeffs[12]*sols(4, m) + coeffs[13], coeffs[9]*sols(3, m), coeffs[11]*sols(3, m),
            coeffs[16]*sols(2, m)*sols(2, m)*sols(4, m) + coeffs[18]*sols(2, m)*sols(4, m) + coeffs[20]*sols(4, m) + coeffs[21], coeffs[17]*sols(3, m), coeffs[19]*sols(3, m);

        Eigen::VectorXd A0(3, 1);
        A0 << -(coeffs[6]*sols(3, m)+coeffs[7]),
            -(coeffs[14]*sols(3, m)+coeffs[15]),
            -(coeffs[22]*sols(3, m)+coeffs[23]);
        Eigen::VectorXd xz = A1.partialPivLu().solve(A0);

        if (xz[0] < 0)
            continue;
        sols(0, m) = std::sqrt(xz[0]); // s x1
        sols(1, m) = xz[2]; // u x2
        sols(3, m) = 1.0 / std::sqrt(V(2, k).real());
        sols(4, m) = 1.0 / std::sqrt(V(3, k).real());
        ++m;
    }

    sols.conservativeResize(5,m);
    return sols;
}


Eigen::MatrixXd varying_focal_monodepth_eigen(Eigen::VectorXd d){
    Eigen::VectorXd coeffs(48);
    coeffs[0] = std::pow(d[8],2) - 2*d[8]*d[9] + std::pow(d[9],2) + std::pow(d[12],2) - 2*d[12]*d[13] + std::pow(d[13],2);
    coeffs[1] = -std::pow(d[0],2) + 2*d[0]*d[1] - std::pow(d[1],2) - std::pow(d[4],2) + 2*d[4]*d[5] - std::pow(d[5],2);
    coeffs[2] = 2*std::pow(d[8],2)*d[20] - 2*d[8]*d[9]*d[20] + 2*std::pow(d[12],2)*d[20] - 2*d[12]*d[13]*d[20] - 2*d[8]*d[9]*d[21] + 2*std::pow(d[9],2)*d[21] - 2*d[12]*d[13]*d[21] + 2*std::pow(d[13],2)*d[21];
    coeffs[3] = -2*std::pow(d[0],2)*d[16] + 2*d[0]*d[1]*d[16] - 2*std::pow(d[4],2)*d[16] + 2*d[4]*d[5]*d[16] + 2*d[0]*d[1]*d[17] - 2*std::pow(d[1],2)*d[17] + 2*d[4]*d[5]*d[17] - 2*std::pow(d[5],2)*d[17];
    coeffs[4] = std::pow(d[8],2)*std::pow(d[20],2) + std::pow(d[12],2)*std::pow(d[20],2) - 2*d[8]*d[9]*d[20]*d[21] - 2*d[12]*d[13]*d[20]*d[21] + std::pow(d[9],2)*std::pow(d[21],2) + std::pow(d[13],2)*std::pow(d[21],2);
    coeffs[5] = std::pow(d[20],2) - 2*d[20]*d[21] + std::pow(d[21],2);
    coeffs[6] = -std::pow(d[0],2)*std::pow(d[16],2) - std::pow(d[4],2)*std::pow(d[16],2) + 2*d[0]*d[1]*d[16]*d[17] + 2*d[4]*d[5]*d[16]*d[17] - std::pow(d[1],2)*std::pow(d[17],2) - std::pow(d[5],2)*std::pow(d[17],2);
    coeffs[7] = -std::pow(d[16],2) + 2*d[16]*d[17] - std::pow(d[17],2);
    coeffs[8] = std::pow(d[8],2) - 2*d[8]*d[10] + std::pow(d[10],2) + std::pow(d[12],2) - 2*d[12]*d[14] + std::pow(d[14],2);
    coeffs[9] = -std::pow(d[0],2) + 2*d[0]*d[2] - std::pow(d[2],2) - std::pow(d[4],2) + 2*d[4]*d[6] - std::pow(d[6],2);
    coeffs[10] = 2*std::pow(d[8],2)*d[20] - 2*d[8]*d[10]*d[20] + 2*std::pow(d[12],2)*d[20] - 2*d[12]*d[14]*d[20] - 2*d[8]*d[10]*d[22] + 2*std::pow(d[10],2)*d[22] - 2*d[12]*d[14]*d[22] + 2*std::pow(d[14],2)*d[22];
    coeffs[11] = -2*std::pow(d[0],2)*d[16] + 2*d[0]*d[2]*d[16] - 2*std::pow(d[4],2)*d[16] + 2*d[4]*d[6]*d[16] + 2*d[0]*d[2]*d[18] - 2*std::pow(d[2],2)*d[18] + 2*d[4]*d[6]*d[18] - 2*std::pow(d[6],2)*d[18];
    coeffs[12] = std::pow(d[8],2)*std::pow(d[20],2) + std::pow(d[12],2)*std::pow(d[20],2) - 2*d[8]*d[10]*d[20]*d[22] - 2*d[12]*d[14]*d[20]*d[22] + std::pow(d[10],2)*std::pow(d[22],2) + std::pow(d[14],2)*std::pow(d[22],2);
    coeffs[13] = std::pow(d[20],2) - 2*d[20]*d[22] + std::pow(d[22],2);
    coeffs[14] = -std::pow(d[0],2)*std::pow(d[16],2) - std::pow(d[4],2)*std::pow(d[16],2) + 2*d[0]*d[2]*d[16]*d[18] + 2*d[4]*d[6]*d[16]*d[18] - std::pow(d[2],2)*std::pow(d[18],2) - std::pow(d[6],2)*std::pow(d[18],2);
    coeffs[15] = -std::pow(d[16],2) + 2*d[16]*d[18] - std::pow(d[18],2);
    coeffs[16] = std::pow(d[8],2) - 2*d[8]*d[11] + std::pow(d[11],2) + std::pow(d[12],2) - 2*d[12]*d[15] + std::pow(d[15],2);
    coeffs[17] = -std::pow(d[0],2) + 2*d[0]*d[3] - std::pow(d[3],2) - std::pow(d[4],2) + 2*d[4]*d[7] - std::pow(d[7],2);
    coeffs[18] = 2*std::pow(d[8],2)*d[20] - 2*d[8]*d[11]*d[20] + 2*std::pow(d[12],2)*d[20] - 2*d[12]*d[15]*d[20] - 2*d[8]*d[11]*d[23] + 2*std::pow(d[11],2)*d[23] - 2*d[12]*d[15]*d[23] + 2*std::pow(d[15],2)*d[23];
    coeffs[19] = -2*std::pow(d[0],2)*d[16] + 2*d[0]*d[3]*d[16] - 2*std::pow(d[4],2)*d[16] + 2*d[4]*d[7]*d[16] + 2*d[0]*d[3]*d[19] - 2*std::pow(d[3],2)*d[19] + 2*d[4]*d[7]*d[19] - 2*std::pow(d[7],2)*d[19];
    coeffs[20] = std::pow(d[8],2)*std::pow(d[20],2) + std::pow(d[12],2)*std::pow(d[20],2) - 2*d[8]*d[11]*d[20]*d[23] - 2*d[12]*d[15]*d[20]*d[23] + std::pow(d[11],2)*std::pow(d[23],2) + std::pow(d[15],2)*std::pow(d[23],2);
    coeffs[21] = std::pow(d[20],2) - 2*d[20]*d[23] + std::pow(d[23],2);
    coeffs[22] = -std::pow(d[0],2)*std::pow(d[16],2) - std::pow(d[4],2)*std::pow(d[16],2) + 2*d[0]*d[3]*d[16]*d[19] + 2*d[4]*d[7]*d[16]*d[19] - std::pow(d[3],2)*std::pow(d[19],2) - std::pow(d[7],2)*std::pow(d[19],2);
    coeffs[23] = -std::pow(d[16],2) + 2*d[16]*d[19] - std::pow(d[19],2);
    coeffs[24] = std::pow(d[9],2) - 2*d[9]*d[10] + std::pow(d[10],2) + std::pow(d[13],2) - 2*d[13]*d[14] + std::pow(d[14],2);
    coeffs[25] = -std::pow(d[1],2) + 2*d[1]*d[2] - std::pow(d[2],2) - std::pow(d[5],2) + 2*d[5]*d[6] - std::pow(d[6],2);
    coeffs[26] = 2*std::pow(d[9],2)*d[21] - 2*d[9]*d[10]*d[21] + 2*std::pow(d[13],2)*d[21] - 2*d[13]*d[14]*d[21] - 2*d[9]*d[10]*d[22] + 2*std::pow(d[10],2)*d[22] - 2*d[13]*d[14]*d[22] + 2*std::pow(d[14],2)*d[22];
    coeffs[27] = -2*std::pow(d[1],2)*d[17] + 2*d[1]*d[2]*d[17] - 2*std::pow(d[5],2)*d[17] + 2*d[5]*d[6]*d[17] + 2*d[1]*d[2]*d[18] - 2*std::pow(d[2],2)*d[18] + 2*d[5]*d[6]*d[18] - 2*std::pow(d[6],2)*d[18];
    coeffs[28] = std::pow(d[9],2)*std::pow(d[21],2) + std::pow(d[13],2)*std::pow(d[21],2) - 2*d[9]*d[10]*d[21]*d[22] - 2*d[13]*d[14]*d[21]*d[22] + std::pow(d[10],2)*std::pow(d[22],2) + std::pow(d[14],2)*std::pow(d[22],2);
    coeffs[29] = std::pow(d[21],2) - 2*d[21]*d[22] + std::pow(d[22],2);
    coeffs[30] = -std::pow(d[1],2)*std::pow(d[17],2) - std::pow(d[5],2)*std::pow(d[17],2) + 2*d[1]*d[2]*d[17]*d[18] + 2*d[5]*d[6]*d[17]*d[18] - std::pow(d[2],2)*std::pow(d[18],2) - std::pow(d[6],2)*std::pow(d[18],2);
    coeffs[31] = -std::pow(d[17],2) + 2*d[17]*d[18] - std::pow(d[18],2);
    coeffs[32] = std::pow(d[9],2) - 2*d[9]*d[11] + std::pow(d[11],2) + std::pow(d[13],2) - 2*d[13]*d[15] + std::pow(d[15],2);
    coeffs[33] = -std::pow(d[1],2) + 2*d[1]*d[3] - std::pow(d[3],2) - std::pow(d[5],2) + 2*d[5]*d[7] - std::pow(d[7],2);
    coeffs[34] = 2*std::pow(d[9],2)*d[21] - 2*d[9]*d[11]*d[21] + 2*std::pow(d[13],2)*d[21] - 2*d[13]*d[15]*d[21] - 2*d[9]*d[11]*d[23] + 2*std::pow(d[11],2)*d[23] - 2*d[13]*d[15]*d[23] + 2*std::pow(d[15],2)*d[23];
    coeffs[35] = -2*std::pow(d[1],2)*d[17] + 2*d[1]*d[3]*d[17] - 2*std::pow(d[5],2)*d[17] + 2*d[5]*d[7]*d[17] + 2*d[1]*d[3]*d[19] - 2*std::pow(d[3],2)*d[19] + 2*d[5]*d[7]*d[19] - 2*std::pow(d[7],2)*d[19];
    coeffs[36] = std::pow(d[9],2)*std::pow(d[21],2) + std::pow(d[13],2)*std::pow(d[21],2) - 2*d[9]*d[11]*d[21]*d[23] - 2*d[13]*d[15]*d[21]*d[23] + std::pow(d[11],2)*std::pow(d[23],2) + std::pow(d[15],2)*std::pow(d[23],2);
    coeffs[37] = std::pow(d[21],2) - 2*d[21]*d[23] + std::pow(d[23],2);
    coeffs[38] = -std::pow(d[1],2)*std::pow(d[17],2) - std::pow(d[5],2)*std::pow(d[17],2) + 2*d[1]*d[3]*d[17]*d[19] + 2*d[5]*d[7]*d[17]*d[19] - std::pow(d[3],2)*std::pow(d[19],2) - std::pow(d[7],2)*std::pow(d[19],2);
    coeffs[39] = -std::pow(d[17],2) + 2*d[17]*d[19] - std::pow(d[19],2);
    coeffs[40] = std::pow(d[10],2) - 2*d[10]*d[11] + std::pow(d[11],2) + std::pow(d[14],2) - 2*d[14]*d[15] + std::pow(d[15],2);
    coeffs[41] = -std::pow(d[2],2) + 2*d[2]*d[3] - std::pow(d[3],2) - std::pow(d[6],2) + 2*d[6]*d[7] - std::pow(d[7],2);
    coeffs[42] = 2*std::pow(d[10],2)*d[22] - 2*d[10]*d[11]*d[22] + 2*std::pow(d[14],2)*d[22] - 2*d[14]*d[15]*d[22] - 2*d[10]*d[11]*d[23] + 2*std::pow(d[11],2)*d[23] - 2*d[14]*d[15]*d[23] + 2*std::pow(d[15],2)*d[23];
    coeffs[43] = -2*std::pow(d[2],2)*d[18] + 2*d[2]*d[3]*d[18] - 2*std::pow(d[6],2)*d[18] + 2*d[6]*d[7]*d[18] + 2*d[2]*d[3]*d[19] - 2*std::pow(d[3],2)*d[19] + 2*d[6]*d[7]*d[19] - 2*std::pow(d[7],2)*d[19];
    coeffs[44] = std::pow(d[10],2)*std::pow(d[22],2) + std::pow(d[14],2)*std::pow(d[22],2) - 2*d[10]*d[11]*d[22]*d[23] - 2*d[14]*d[15]*d[22]*d[23] + std::pow(d[11],2)*std::pow(d[23],2) + std::pow(d[15],2)*std::pow(d[23],2);
    coeffs[45] = std::pow(d[22],2) - 2*d[22]*d[23] + std::pow(d[23],2);
    coeffs[46] = -std::pow(d[2],2)*std::pow(d[18],2) - std::pow(d[6],2)*std::pow(d[18],2) + 2*d[2]*d[3]*d[18]*d[19] + 2*d[6]*d[7]*d[18]*d[19] - std::pow(d[3],2)*std::pow(d[19],2) - std::pow(d[7],2)*std::pow(d[19],2);
    coeffs[47] = -std::pow(d[18],2) + 2*d[18]*d[19] - std::pow(d[19],2);


    Eigen::MatrixXd C0(6, 6);
    C0 << coeffs[1], coeffs[3], coeffs[4], coeffs[5], coeffs[6], coeffs[7],
        coeffs[9], coeffs[11], coeffs[12], coeffs[13], coeffs[14], coeffs[15],
        coeffs[17], coeffs[19], coeffs[20], coeffs[21], coeffs[22], coeffs[23],
        coeffs[25], coeffs[27], coeffs[28], coeffs[29], coeffs[30], coeffs[31],
        coeffs[33], coeffs[35], coeffs[36], coeffs[37], coeffs[38], coeffs[39],
        coeffs[41], coeffs[43], coeffs[44], coeffs[45], coeffs[46], coeffs[47];


    Eigen::MatrixXd C1(6, 2);
    C1 << coeffs[0], coeffs[2],
        coeffs[8], coeffs[10],
        coeffs[16], coeffs[18],
        coeffs[24], coeffs[26],
        coeffs[32], coeffs[34],
        coeffs[40], coeffs[42];

    Eigen::MatrixXd C12 = -C0.partialPivLu().solve(C1);
    Eigen::MatrixXd AM(2, 2);
    AM << 0.0, 1.0,
        C12(2,0), C12(2,1);

    Eigen::EigenSolver<Eigen::MatrixXd> es(AM);
    Eigen::ArrayXcd D = es.eigenvalues();

    Eigen::MatrixXd sols(5, 2);
    int m = 0;
    for (int k = 0; k < 2; ++k)
    {

        if (abs(D(k).imag()) > 0.001)
            continue;

        sols(2, m) = 1.0 / D(k).real(); // v x3
        Eigen::MatrixXd A1(5, 5);
        double v2 = sols(2, m)*sols(2, m);
        A1 << coeffs[1], coeffs[3], coeffs[0]*v2+coeffs[2]*sols(2, m)+coeffs[4], coeffs[5], coeffs[6],
            coeffs[9], coeffs[11], coeffs[8]*v2+coeffs[10]*sols(2, m)+coeffs[12], coeffs[13], coeffs[14],
            coeffs[17], coeffs[19], coeffs[16]*v2+coeffs[18]*sols(2, m)+coeffs[20], coeffs[21], coeffs[22],
            coeffs[25], coeffs[27], coeffs[24]*v2+coeffs[26]*sols(2, m)+coeffs[28], coeffs[29], coeffs[30],
            coeffs[33], coeffs[35], coeffs[32]*v2+coeffs[34]*sols(2, m)+coeffs[36], coeffs[37], coeffs[38];
        Eigen::VectorXd A0(5, 1);
        A0 << -coeffs[7],
            -coeffs[15],
            -coeffs[23],
            -coeffs[31],
            -coeffs[39];
        Eigen::VectorXd xz = A1.partialPivLu().solve(A0);

        if (xz[3] < 0.0 || xz[4] < 0.0 || xz[2]/xz[3] < 0.0)
            continue;

        sols(0, m) = std::sqrt(xz[3]); // s x1
        sols(3, m) = 1.0 / std::sqrt(xz[4]); // f x4
        sols(1, m) = xz[1]/xz[4]; // u x2
        sols(4, m) = 1.0 / std::sqrt(xz[2]/xz[3]); // m x5
        ++m;
    }

    sols.conservativeResize(5,m);
    return sols;
}


void varying_focal_monodepth_relpose_ours(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                                          const std::vector<Eigen::Vector2d> &sigma, bool use_eigen,
                                          std::vector<ImagePair> *models){
    models->clear();
    models->reserve(4);
    std::vector<Eigen::Vector3d> x1h(4);
    std::vector<Eigen::Vector3d> x2h(4);
    for (int i = 0; i < 4; ++i) {
        x1h[i] = x1[i].homogeneous();
        x2h[i] = x2[i].homogeneous();
    }

    double depth1[4];
    double depth2[4];
    for (int i = 0; i < 4; ++i) {
        depth1[i] = sigma[i][0];
        depth2[i] = sigma[i][1];
    }

    Eigen::VectorXd datain(24);
    datain << x1h[0][0], x1h[1][0], x1h[2][0], x1h[3][0], x1h[0][1], x1h[1][1], x1h[2][1], x1h[3][1], x2h[0][0],
        x2h[1][0], x2h[2][0], x2h[3][0], x2h[0][1], x2h[1][1], x2h[2][1], x2h[3][1], depth1[0], depth1[1], depth1[2],
        depth1[3], depth2[0], depth2[1], depth2[2], depth1[3];

    Eigen::MatrixXd sols;
    if (use_eigen)
        sols = varying_focal_monodepth_eigen(datain);
    else
        sols = varying_focal_monodepth_gj(datain);

    models->reserve(sols.cols());

    for (int k = 0; k < sols.cols(); ++k){
        double s = sols(0, k);
        double u = sols(1, k);
        double v = sols(2, k);
        double f = sols(3, k);
        double w = sols(4, k);

        Eigen::Matrix3d K1inv;
        K1inv << 1.0 / f, 0, 0,
            0, 1.0 / f, 0,
            0, 0, 1;

        Eigen::Matrix3d K2inv;
        K2inv << 1.0 / w, 0, 0,
            0, 1.0 / w, 0,
            0, 0, 1;

        Eigen::Vector3d v1 = s * (depth2[0] + v) * K2inv*x2h[0] - s * (depth2[1] + v) * K2inv*x2h[1];
        Eigen::Vector3d v2 = s * (depth2[0] + v) * K2inv*x2h[0] - s * (depth2[2] + v) * K2inv*x2h[2];
        Eigen::Matrix3d Y;
        Y << v1, v2, v1.cross(v2);

        Eigen::Vector3d u1 = (depth1[0] + u) * K1inv*x1h[0] - (depth1[1] + u) * K1inv*x1h[1];
        Eigen::Vector3d u2 = (depth1[0] + u) * K1inv*x1h[0] - (depth1[2] + u) * K1inv*x1h[2];
        Eigen::Matrix3d X;
        X << u1, u2, u1.cross(u2);
        X = X.inverse().eval();

        Eigen::Matrix3d rot = Y * X;

        Eigen::Vector3d trans1 = (depth1[0] + u) * rot * K1inv*x1h[0];
        Eigen::Vector3d trans2 = s * (depth2[0] + v) * K2inv*x2h[0];
        Eigen::Vector3d trans = trans2 - trans1;

        CameraPose pose = CameraPose(rot, trans);
        Camera camera1 = Camera("SIMPLE_PINHOLE", std::vector<double>{f, 0.0, 0.0}, -1, -1);
        Camera camera2 = Camera("SIMPLE_PINHOLE", std::vector<double>{w, 0.0, 0.0}, -1, -1);
        models->emplace_back(pose, camera1, camera2);
    }
}
}