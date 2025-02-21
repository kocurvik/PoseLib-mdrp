//
// Created by kocur on 19-Nov-24.
//

#include "shared_focal_monodepth_relpose_eigen.h"

#include "PoseLib/misc/univariate.h"
namespace poselib {

Eigen::MatrixXd shared_focal_monodepth_relpose_gb_impl(Eigen::VectorXd d) {
    Eigen::VectorXd coeffs(32);
    coeffs[0] = std::pow(d[8],2) - 2*d[8]*d[9] + std::pow(d[9],2) + std::pow(d[12],2) - 2*d[12]*d[13] + std::pow(d[13],2);
    coeffs[1] = 2*std::pow(d[8],2)*d[20] - 2*d[8]*d[9]*d[20] + 2*std::pow(d[12],2)*d[20] - 2*d[12]*d[13]*d[20] - 2*d[8]*d[9]*d[21] + 2*std::pow(d[9],2)*d[21] - 2*d[12]*d[13]*d[21] + 2*std::pow(d[13],2)*d[21];
    coeffs[2] = -std::pow(d[0],2) + 2*d[0]*d[1] - std::pow(d[1],2) - std::pow(d[4],2) + 2*d[4]*d[5] - std::pow(d[5],2);
    coeffs[3] = std::pow(d[20],2) - 2*d[20]*d[21] + std::pow(d[21],2);
    coeffs[4] = std::pow(d[8],2)*std::pow(d[20],2) + std::pow(d[12],2)*std::pow(d[20],2) - 2*d[8]*d[9]*d[20]*d[21] - 2*d[12]*d[13]*d[20]*d[21] + std::pow(d[9],2)*std::pow(d[21],2) + std::pow(d[13],2)*std::pow(d[21],2);
    coeffs[5] = -2*std::pow(d[0],2)*d[16] + 2*d[0]*d[1]*d[16] - 2*std::pow(d[4],2)*d[16] + 2*d[4]*d[5]*d[16] + 2*d[0]*d[1]*d[17] - 2*std::pow(d[1],2)*d[17] + 2*d[4]*d[5]*d[17] - 2*std::pow(d[5],2)*d[17];
    coeffs[6] = -std::pow(d[16],2) + 2*d[16]*d[17] - std::pow(d[17],2);
    coeffs[7] = -std::pow(d[0],2)*std::pow(d[16],2) - std::pow(d[4],2)*std::pow(d[16],2) + 2*d[0]*d[1]*d[16]*d[17] + 2*d[4]*d[5]*d[16]*d[17] - std::pow(d[1],2)*std::pow(d[17],2) - std::pow(d[5],2)*std::pow(d[17],2);
    coeffs[8] = std::pow(d[8],2) - 2*d[8]*d[10] + std::pow(d[10],2) + std::pow(d[12],2) - 2*d[12]*d[14] + std::pow(d[14],2);
    coeffs[9] = 2*std::pow(d[8],2)*d[20] - 2*d[8]*d[10]*d[20] + 2*std::pow(d[12],2)*d[20] - 2*d[12]*d[14]*d[20] - 2*d[8]*d[10]*d[22] + 2*std::pow(d[10],2)*d[22] - 2*d[12]*d[14]*d[22] + 2*std::pow(d[14],2)*d[22];
    coeffs[10] = -std::pow(d[0],2) + 2*d[0]*d[2] - std::pow(d[2],2) - std::pow(d[4],2) + 2*d[4]*d[6] - std::pow(d[6],2);
    coeffs[11] = std::pow(d[20],2) - 2*d[20]*d[22] + std::pow(d[22],2);
    coeffs[12] = std::pow(d[8],2)*std::pow(d[20],2) + std::pow(d[12],2)*std::pow(d[20],2) - 2*d[8]*d[10]*d[20]*d[22] - 2*d[12]*d[14]*d[20]*d[22] + std::pow(d[10],2)*std::pow(d[22],2) + std::pow(d[14],2)*std::pow(d[22],2);
    coeffs[13] = -2*std::pow(d[0],2)*d[16] + 2*d[0]*d[2]*d[16] - 2*std::pow(d[4],2)*d[16] + 2*d[4]*d[6]*d[16] + 2*d[0]*d[2]*d[18] - 2*std::pow(d[2],2)*d[18] + 2*d[4]*d[6]*d[18] - 2*std::pow(d[6],2)*d[18];
    coeffs[14] = -std::pow(d[16],2) + 2*d[16]*d[18] - std::pow(d[18],2);
    coeffs[15] = -std::pow(d[0],2)*std::pow(d[16],2) - std::pow(d[4],2)*std::pow(d[16],2) + 2*d[0]*d[2]*d[16]*d[18] + 2*d[4]*d[6]*d[16]*d[18] - std::pow(d[2],2)*std::pow(d[18],2) - std::pow(d[6],2)*std::pow(d[18],2);
    coeffs[16] = std::pow(d[8],2) - 2*d[8]*d[11] + std::pow(d[11],2) + std::pow(d[12],2) - 2*d[12]*d[15] + std::pow(d[15],2);
    coeffs[17] = 2*std::pow(d[8],2)*d[20] - 2*d[8]*d[11]*d[20] + 2*std::pow(d[12],2)*d[20] - 2*d[12]*d[15]*d[20] - 2*d[8]*d[11]*d[23] + 2*std::pow(d[11],2)*d[23] - 2*d[12]*d[15]*d[23] + 2*std::pow(d[15],2)*d[23];
    coeffs[18] = -std::pow(d[0],2) + 2*d[0]*d[3] - std::pow(d[3],2) - std::pow(d[4],2) + 2*d[4]*d[7] - std::pow(d[7],2);
    coeffs[19] = std::pow(d[20],2) - 2*d[20]*d[23] + std::pow(d[23],2);
    coeffs[20] = std::pow(d[8],2)*std::pow(d[20],2) + std::pow(d[12],2)*std::pow(d[20],2) - 2*d[8]*d[11]*d[20]*d[23] - 2*d[12]*d[15]*d[20]*d[23] + std::pow(d[11],2)*std::pow(d[23],2) + std::pow(d[15],2)*std::pow(d[23],2);
    coeffs[21] = -2*std::pow(d[0],2)*d[16] + 2*d[0]*d[3]*d[16] - 2*std::pow(d[4],2)*d[16] + 2*d[4]*d[7]*d[16] + 2*d[0]*d[3]*d[19] - 2*std::pow(d[3],2)*d[19] + 2*d[4]*d[7]*d[19] - 2*std::pow(d[7],2)*d[19];
    coeffs[22] = -std::pow(d[16],2) + 2*d[16]*d[19] - std::pow(d[19],2);
    coeffs[23] = -std::pow(d[0],2)*std::pow(d[16],2) - std::pow(d[4],2)*std::pow(d[16],2) + 2*d[0]*d[3]*d[16]*d[19] + 2*d[4]*d[7]*d[16]*d[19] - std::pow(d[3],2)*std::pow(d[19],2) - std::pow(d[7],2)*std::pow(d[19],2);
    coeffs[24] = std::pow(d[9],2) - 2*d[9]*d[10] + std::pow(d[10],2) + std::pow(d[13],2) - 2*d[13]*d[14] + std::pow(d[14],2);
    coeffs[25] = 2*std::pow(d[9],2)*d[21] - 2*d[9]*d[10]*d[21] + 2*std::pow(d[13],2)*d[21] - 2*d[13]*d[14]*d[21] - 2*d[9]*d[10]*d[22] + 2*std::pow(d[10],2)*d[22] - 2*d[13]*d[14]*d[22] + 2*std::pow(d[14],2)*d[22];
    coeffs[26] = -std::pow(d[1],2) + 2*d[1]*d[2] - std::pow(d[2],2) - std::pow(d[5],2) + 2*d[5]*d[6] - std::pow(d[6],2);
    coeffs[27] = std::pow(d[21],2) - 2*d[21]*d[22] + std::pow(d[22],2);
    coeffs[28] = std::pow(d[9],2)*std::pow(d[21],2) + std::pow(d[13],2)*std::pow(d[21],2) - 2*d[9]*d[10]*d[21]*d[22] - 2*d[13]*d[14]*d[21]*d[22] + std::pow(d[10],2)*std::pow(d[22],2) + std::pow(d[14],2)*std::pow(d[22],2);
    coeffs[29] = -2*std::pow(d[1],2)*d[17] + 2*d[1]*d[2]*d[17] - 2*std::pow(d[5],2)*d[17] + 2*d[5]*d[6]*d[17] + 2*d[1]*d[2]*d[18] - 2*std::pow(d[2],2)*d[18] + 2*d[5]*d[6]*d[18] - 2*std::pow(d[6],2)*d[18];
    coeffs[30] = -std::pow(d[17],2) + 2*d[17]*d[18] - std::pow(d[18],2);
    coeffs[31] = -std::pow(d[1],2)*std::pow(d[17],2) - std::pow(d[5],2)*std::pow(d[17],2) + 2*d[1]*d[2]*d[17]*d[18] + 2*d[5]*d[6]*d[17]*d[18] - std::pow(d[2],2)*std::pow(d[18],2) - std::pow(d[6],2)*std::pow(d[18],2);


    static const int coeffs_ind[] = {0,8,16,24,0,8,16,24,0,8,16,24,1,0,9,8,17,16,25,24,2,10,18,26,2,10,18,26,5,2,13,10,21,18,29,26,2,10,18,26,6,5,14,13,21,22,29,30,2,18,
                                     10,26,3,19,11,27,6,14,22,30,7,5,15,13,23,21,31,29,5,13,2,21,18,10,26,29,4,20,3,19,12,11,27,28,6,14,5,21,22,13,29,30,4,20,12,28,7,15,5,23,
                                     21,13,29,31,1,9,17,0,16,8,25,24,3,11,19,27,3,11,19,27,3,11,19,27,6,22,14,30,7,6,15,14,23,22,31,30,1,9,17,0,16,8,24,25,4,1,12,9,20,17,
                                     28,25,4,3,12,11,20,1,17,19,9,28,25,27,4,12,20,1,17,9,25,28,4,12,20,28,7,23,6,22,15,14,30,31,7,23,15,31,7,15,23,31};

    static const int C_ind[] = {0,4,11,20,25,29,32,43,50,54,60,71,72,75,76,79,83,86,92,94,96,100,107,116,121,125,128,139,144,146,148,150,155,156,164,167,171,175,182,190,192,193,196,197,200,203,211,212,225,226,
                                232,237,249,250,256,261,265,269,272,283,288,290,292,294,299,300,308,311,315,319,325,326,327,329,330,334,345,346,349,351,352,353,354,357,363,367,369,370,374,376,381,382,397,399,401,402,411,415,421,422,
                                423,425,426,430,433,437,440,441,442,448,451,453,456,460,467,476,481,485,488,499,507,511,518,526,537,538,544,549,553,554,557,558,560,564,571,575,578,582,588,589,591,593,594,599,600,603,604,607,611,614,
                                620,622,625,626,629,630,632,633,634,636,640,643,645,647,650,654,660,661,663,665,666,671,675,679,686,694,705,706,709,711,712,713,714,717,733,735,737,738,746,750,756,767};

    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(24,32);
    for (int i = 0; i < 192; i++) {
        C(C_ind[i]) = coeffs(coeffs_ind[i]);
    }

    Eigen::MatrixXd C0 = C.leftCols(24);
    Eigen::MatrixXd C1 = C.rightCols(8);
    Eigen::MatrixXd C12 = C0.partialPivLu().solve(C1);
    Eigen::MatrixXd RR(14, 8);
    RR << -C12.bottomRows(6), Eigen::MatrixXd::Identity(8, 8);

    static const int AM_ind[] = {0,1,2,8,3,4,11,5};
    Eigen::MatrixXd AM(8, 8);
    for (int i = 0; i < 8; i++) {
        AM.row(i) = RR.row(AM_ind[i]);
    }





    Eigen::EigenSolver<Eigen::MatrixXd> es(AM);
    Eigen::ArrayXcd D = es.eigenvalues();
    Eigen::ArrayXXcd V = es.eigenvectors();
    V = (V / V.row(6).replicate(8, 1)).eval();

    Eigen::MatrixXd sols(4, 8);
    int m = 0;
    for (int k = 0; k < 8; ++k)
    {

        if (abs(D(k).imag()) > 0.001 || D(k).real() < 0.0 ||
            abs(V(7, k).imag()) > 0.001)
            continue;

        sols(3, m) = std::sqrt(D(k).real()); // f
        sols(2, m) = V(7, k).real(); // v
        double v2 = sols(2, m)*sols(2, m);
        Eigen::MatrixXd A1(3, 3);
        A1 << coeffs[0]*v2+coeffs[1]*sols(2, m)+coeffs[3]*D(k).real()+coeffs[4], coeffs[2], coeffs[5],
            coeffs[8]*v2+coeffs[9]*sols(2, m)+coeffs[11]*D(k).real()+coeffs[12], coeffs[10], coeffs[13],
            coeffs[16]*v2+coeffs[17]*sols(2, m)+coeffs[19]*D(k).real()+coeffs[20], coeffs[18], coeffs[21];
        Eigen::VectorXd A0(3, 1);
        A0 << -(coeffs[6]*D(k).real()+coeffs[7]),
            -(coeffs[14]*D(k).real()+coeffs[15]),
            -(coeffs[22]*D(k).real()+coeffs[23]);
        Eigen::VectorXd xz = A1.partialPivLu().solve(A0);
        if (xz[0] < 0)
            continue;
        sols(0, m) = std::sqrt(xz[0]); // s
        sols(1, m) = xz[2]; // u
        ++m;
    }

    sols.conservativeResize(4,m);
    return sols;
}

Eigen::MatrixXd shared_focal_monodepth_relpose_eigen_impl(Eigen::VectorXd d) {
    Eigen::VectorXd coeffs(48);
    coeffs[0] = std::pow(d[8], 2) - 2 * d[8] * d[9] + std::pow(d[9], 2) + std::pow(d[12], 2) - 2 * d[12] * d[13] +
                std::pow(d[13], 2);
    coeffs[1] = -std::pow(d[0], 2) + 2 * d[0] * d[1] - std::pow(d[1], 2) - std::pow(d[4], 2) + 2 * d[4] * d[5] -
                std::pow(d[5], 2);
    coeffs[2] = 2 * std::pow(d[8], 2) * d[20] - 2 * d[8] * d[9] * d[20] + 2 * std::pow(d[12], 2) * d[20] -
                2 * d[12] * d[13] * d[20] - 2 * d[8] * d[9] * d[21] + 2 * std::pow(d[9], 2) * d[21] -
                2 * d[12] * d[13] * d[21] + 2 * std::pow(d[13], 2) * d[21];
    coeffs[3] = std::pow(d[20], 2) - 2 * d[20] * d[21] + std::pow(d[21], 2);
    coeffs[4] = std::pow(d[8], 2) * std::pow(d[20], 2) + std::pow(d[12], 2) * std::pow(d[20], 2) -
                2 * d[8] * d[9] * d[20] * d[21] - 2 * d[12] * d[13] * d[20] * d[21] +
                std::pow(d[9], 2) * std::pow(d[21], 2) + std::pow(d[13], 2) * std::pow(d[21], 2);
    coeffs[5] = -2 * std::pow(d[0], 2) * d[16] + 2 * d[0] * d[1] * d[16] - 2 * std::pow(d[4], 2) * d[16] +
                2 * d[4] * d[5] * d[16] + 2 * d[0] * d[1] * d[17] - 2 * std::pow(d[1], 2) * d[17] +
                2 * d[4] * d[5] * d[17] - 2 * std::pow(d[5], 2) * d[17];
    coeffs[6] = -std::pow(d[16], 2) + 2 * d[16] * d[17] - std::pow(d[17], 2);
    coeffs[7] = -std::pow(d[0], 2) * std::pow(d[16], 2) - std::pow(d[4], 2) * std::pow(d[16], 2) +
                2 * d[0] * d[1] * d[16] * d[17] + 2 * d[4] * d[5] * d[16] * d[17] -
                std::pow(d[1], 2) * std::pow(d[17], 2) - std::pow(d[5], 2) * std::pow(d[17], 2);
    coeffs[8] = std::pow(d[8], 2) - 2 * d[8] * d[10] + std::pow(d[10], 2) + std::pow(d[12], 2) - 2 * d[12] * d[14] +
                std::pow(d[14], 2);
    coeffs[9] = -std::pow(d[0], 2) + 2 * d[0] * d[2] - std::pow(d[2], 2) - std::pow(d[4], 2) + 2 * d[4] * d[6] -
                std::pow(d[6], 2);
    coeffs[10] = 2 * std::pow(d[8], 2) * d[20] - 2 * d[8] * d[10] * d[20] + 2 * std::pow(d[12], 2) * d[20] -
                 2 * d[12] * d[14] * d[20] - 2 * d[8] * d[10] * d[22] + 2 * std::pow(d[10], 2) * d[22] -
                 2 * d[12] * d[14] * d[22] + 2 * std::pow(d[14], 2) * d[22];
    coeffs[11] = std::pow(d[20], 2) - 2 * d[20] * d[22] + std::pow(d[22], 2);
    coeffs[12] = std::pow(d[8], 2) * std::pow(d[20], 2) + std::pow(d[12], 2) * std::pow(d[20], 2) -
                 2 * d[8] * d[10] * d[20] * d[22] - 2 * d[12] * d[14] * d[20] * d[22] +
                 std::pow(d[10], 2) * std::pow(d[22], 2) + std::pow(d[14], 2) * std::pow(d[22], 2);
    coeffs[13] = -2 * std::pow(d[0], 2) * d[16] + 2 * d[0] * d[2] * d[16] - 2 * std::pow(d[4], 2) * d[16] +
                 2 * d[4] * d[6] * d[16] + 2 * d[0] * d[2] * d[18] - 2 * std::pow(d[2], 2) * d[18] +
                 2 * d[4] * d[6] * d[18] - 2 * std::pow(d[6], 2) * d[18];
    coeffs[14] = -std::pow(d[16], 2) + 2 * d[16] * d[18] - std::pow(d[18], 2);
    coeffs[15] = -std::pow(d[0], 2) * std::pow(d[16], 2) - std::pow(d[4], 2) * std::pow(d[16], 2) +
                 2 * d[0] * d[2] * d[16] * d[18] + 2 * d[4] * d[6] * d[16] * d[18] -
                 std::pow(d[2], 2) * std::pow(d[18], 2) - std::pow(d[6], 2) * std::pow(d[18], 2);
    coeffs[16] = std::pow(d[8], 2) - 2 * d[8] * d[11] + std::pow(d[11], 2) + std::pow(d[12], 2) - 2 * d[12] * d[15] +
                 std::pow(d[15], 2);
    coeffs[17] = -std::pow(d[0], 2) + 2 * d[0] * d[3] - std::pow(d[3], 2) - std::pow(d[4], 2) + 2 * d[4] * d[7] -
                 std::pow(d[7], 2);
    coeffs[18] = 2 * std::pow(d[8], 2) * d[20] - 2 * d[8] * d[11] * d[20] + 2 * std::pow(d[12], 2) * d[20] -
                 2 * d[12] * d[15] * d[20] - 2 * d[8] * d[11] * d[23] + 2 * std::pow(d[11], 2) * d[23] -
                 2 * d[12] * d[15] * d[23] + 2 * std::pow(d[15], 2) * d[23];
    coeffs[19] = std::pow(d[20], 2) - 2 * d[20] * d[23] + std::pow(d[23], 2);
    coeffs[20] = std::pow(d[8], 2) * std::pow(d[20], 2) + std::pow(d[12], 2) * std::pow(d[20], 2) -
                 2 * d[8] * d[11] * d[20] * d[23] - 2 * d[12] * d[15] * d[20] * d[23] +
                 std::pow(d[11], 2) * std::pow(d[23], 2) + std::pow(d[15], 2) * std::pow(d[23], 2);
    coeffs[21] = -2 * std::pow(d[0], 2) * d[16] + 2 * d[0] * d[3] * d[16] - 2 * std::pow(d[4], 2) * d[16] +
                 2 * d[4] * d[7] * d[16] + 2 * d[0] * d[3] * d[19] - 2 * std::pow(d[3], 2) * d[19] +
                 2 * d[4] * d[7] * d[19] - 2 * std::pow(d[7], 2) * d[19];
    coeffs[22] = -std::pow(d[16], 2) + 2 * d[16] * d[19] - std::pow(d[19], 2);
    coeffs[23] = -std::pow(d[0], 2) * std::pow(d[16], 2) - std::pow(d[4], 2) * std::pow(d[16], 2) +
                 2 * d[0] * d[3] * d[16] * d[19] + 2 * d[4] * d[7] * d[16] * d[19] -
                 std::pow(d[3], 2) * std::pow(d[19], 2) - std::pow(d[7], 2) * std::pow(d[19], 2);
    coeffs[24] = std::pow(d[9], 2) - 2 * d[9] * d[10] + std::pow(d[10], 2) + std::pow(d[13], 2) - 2 * d[13] * d[14] +
                 std::pow(d[14], 2);
    coeffs[25] = -std::pow(d[1], 2) + 2 * d[1] * d[2] - std::pow(d[2], 2) - std::pow(d[5], 2) + 2 * d[5] * d[6] -
                 std::pow(d[6], 2);
    coeffs[26] = 2 * std::pow(d[9], 2) * d[21] - 2 * d[9] * d[10] * d[21] + 2 * std::pow(d[13], 2) * d[21] -
                 2 * d[13] * d[14] * d[21] - 2 * d[9] * d[10] * d[22] + 2 * std::pow(d[10], 2) * d[22] -
                 2 * d[13] * d[14] * d[22] + 2 * std::pow(d[14], 2) * d[22];
    coeffs[27] = std::pow(d[21], 2) - 2 * d[21] * d[22] + std::pow(d[22], 2);
    coeffs[28] = std::pow(d[9], 2) * std::pow(d[21], 2) + std::pow(d[13], 2) * std::pow(d[21], 2) -
                 2 * d[9] * d[10] * d[21] * d[22] - 2 * d[13] * d[14] * d[21] * d[22] +
                 std::pow(d[10], 2) * std::pow(d[22], 2) + std::pow(d[14], 2) * std::pow(d[22], 2);
    coeffs[29] = -2 * std::pow(d[1], 2) * d[17] + 2 * d[1] * d[2] * d[17] - 2 * std::pow(d[5], 2) * d[17] +
                 2 * d[5] * d[6] * d[17] + 2 * d[1] * d[2] * d[18] - 2 * std::pow(d[2], 2) * d[18] +
                 2 * d[5] * d[6] * d[18] - 2 * std::pow(d[6], 2) * d[18];
    coeffs[30] = -std::pow(d[17], 2) + 2 * d[17] * d[18] - std::pow(d[18], 2);
    coeffs[31] = -std::pow(d[1], 2) * std::pow(d[17], 2) - std::pow(d[5], 2) * std::pow(d[17], 2) +
                 2 * d[1] * d[2] * d[17] * d[18] + 2 * d[5] * d[6] * d[17] * d[18] -
                 std::pow(d[2], 2) * std::pow(d[18], 2) - std::pow(d[6], 2) * std::pow(d[18], 2);
    coeffs[32] = std::pow(d[9], 2) - 2 * d[9] * d[11] + std::pow(d[11], 2) + std::pow(d[13], 2) - 2 * d[13] * d[15] +
                 std::pow(d[15], 2);
    coeffs[33] = -std::pow(d[1], 2) + 2 * d[1] * d[3] - std::pow(d[3], 2) - std::pow(d[5], 2) + 2 * d[5] * d[7] -
                 std::pow(d[7], 2);
    coeffs[34] = 2 * std::pow(d[9], 2) * d[21] - 2 * d[9] * d[11] * d[21] + 2 * std::pow(d[13], 2) * d[21] -
                 2 * d[13] * d[15] * d[21] - 2 * d[9] * d[11] * d[23] + 2 * std::pow(d[11], 2) * d[23] -
                 2 * d[13] * d[15] * d[23] + 2 * std::pow(d[15], 2) * d[23];
    coeffs[35] = std::pow(d[21], 2) - 2 * d[21] * d[23] + std::pow(d[23], 2);
    coeffs[36] = std::pow(d[9], 2) * std::pow(d[21], 2) + std::pow(d[13], 2) * std::pow(d[21], 2) -
                 2 * d[9] * d[11] * d[21] * d[23] - 2 * d[13] * d[15] * d[21] * d[23] +
                 std::pow(d[11], 2) * std::pow(d[23], 2) + std::pow(d[15], 2) * std::pow(d[23], 2);
    coeffs[37] = -2 * std::pow(d[1], 2) * d[17] + 2 * d[1] * d[3] * d[17] - 2 * std::pow(d[5], 2) * d[17] +
                 2 * d[5] * d[7] * d[17] + 2 * d[1] * d[3] * d[19] - 2 * std::pow(d[3], 2) * d[19] +
                 2 * d[5] * d[7] * d[19] - 2 * std::pow(d[7], 2) * d[19];
    coeffs[38] = -std::pow(d[17], 2) + 2 * d[17] * d[19] - std::pow(d[19], 2);
    coeffs[39] = -std::pow(d[1], 2) * std::pow(d[17], 2) - std::pow(d[5], 2) * std::pow(d[17], 2) +
                 2 * d[1] * d[3] * d[17] * d[19] + 2 * d[5] * d[7] * d[17] * d[19] -
                 std::pow(d[3], 2) * std::pow(d[19], 2) - std::pow(d[7], 2) * std::pow(d[19], 2);
    coeffs[40] = std::pow(d[10], 2) - 2 * d[10] * d[11] + std::pow(d[11], 2) + std::pow(d[14], 2) - 2 * d[14] * d[15] +
                 std::pow(d[15], 2);
    coeffs[41] = -std::pow(d[2], 2) + 2 * d[2] * d[3] - std::pow(d[3], 2) - std::pow(d[6], 2) + 2 * d[6] * d[7] -
                 std::pow(d[7], 2);
    coeffs[42] = 2 * std::pow(d[10], 2) * d[22] - 2 * d[10] * d[11] * d[22] + 2 * std::pow(d[14], 2) * d[22] -
                 2 * d[14] * d[15] * d[22] - 2 * d[10] * d[11] * d[23] + 2 * std::pow(d[11], 2) * d[23] -
                 2 * d[14] * d[15] * d[23] + 2 * std::pow(d[15], 2) * d[23];
    coeffs[43] = std::pow(d[22], 2) - 2 * d[22] * d[23] + std::pow(d[23], 2);
    coeffs[44] = std::pow(d[10], 2) * std::pow(d[22], 2) + std::pow(d[14], 2) * std::pow(d[22], 2) -
                 2 * d[10] * d[11] * d[22] * d[23] - 2 * d[14] * d[15] * d[22] * d[23] +
                 std::pow(d[11], 2) * std::pow(d[23], 2) + std::pow(d[15], 2) * std::pow(d[23], 2);
    coeffs[45] = -2 * std::pow(d[2], 2) * d[18] + 2 * d[2] * d[3] * d[18] - 2 * std::pow(d[6], 2) * d[18] +
                 2 * d[6] * d[7] * d[18] + 2 * d[2] * d[3] * d[19] - 2 * std::pow(d[3], 2) * d[19] +
                 2 * d[6] * d[7] * d[19] - 2 * std::pow(d[7], 2) * d[19];
    coeffs[46] = -std::pow(d[18], 2) + 2 * d[18] * d[19] - std::pow(d[19], 2);
    coeffs[47] = -std::pow(d[2], 2) * std::pow(d[18], 2) - std::pow(d[6], 2) * std::pow(d[18], 2) +
                 2 * d[2] * d[3] * d[18] * d[19] + 2 * d[6] * d[7] * d[18] * d[19] -
                 std::pow(d[3], 2) * std::pow(d[19], 2) - std::pow(d[7], 2) * std::pow(d[19], 2);

    Eigen::MatrixXd C0(6, 6);
    C0 << coeffs[0], coeffs[1], coeffs[2], coeffs[4], coeffs[5], coeffs[7], coeffs[8], coeffs[9], coeffs[10],
        coeffs[12], coeffs[13], coeffs[15], coeffs[16], coeffs[17], coeffs[18], coeffs[20], coeffs[21], coeffs[23],
        coeffs[24], coeffs[25], coeffs[26], coeffs[28], coeffs[29], coeffs[31], coeffs[32], coeffs[33], coeffs[34],
        coeffs[36], coeffs[37], coeffs[39], coeffs[40], coeffs[41], coeffs[42], coeffs[44], coeffs[45], coeffs[47];

    Eigen::MatrixXd C1(6, 2);
    C1 << coeffs[3], coeffs[6], coeffs[11], coeffs[14], coeffs[19], coeffs[22], coeffs[27], coeffs[30], coeffs[35],
        coeffs[38], coeffs[43], coeffs[46];

    Eigen::MatrixXd C12 = -C0.partialPivLu().solve(C1);
    Eigen::MatrixXd AM(2, 2);
    AM << C12(3, 0), C12(3, 1), C12(5, 0), C12(5, 1);

    Eigen::EigenSolver<Eigen::Matrix<double, 2, 2>> es(AM);
    Eigen::ArrayXcd D = es.eigenvalues();
    Eigen::ArrayXXcd V = es.eigenvectors();

    Eigen::MatrixXcd V0(1, 2);
    V0 = V.row(0) / V.row(1);

    Eigen::MatrixXd sols(4, 2);
    int m = 0;
    for (int k = 0; k < 2; ++k) {

        if (abs(D(k).imag()) > 0.001 || D(k).real() < 0.0 || abs(V0(0, k).imag()) > 0.001 || V0(0, k).real() < 0.0)
            continue;

        //         sols(0, m) = 1.0 / std::sqrt(D(k).real()); // f w
        double cc = V0(0, k).real(); // c x
        Eigen::MatrixXd A1(4, 4);
        A1 << coeffs[0] * cc, coeffs[1], coeffs[2] * cc, coeffs[5], coeffs[8] * cc, coeffs[9], coeffs[10] * cc,
            coeffs[13], coeffs[16] * cc, coeffs[17], coeffs[18] * cc, coeffs[21], coeffs[24] * cc, coeffs[25],
            coeffs[26] * cc, coeffs[29];
        Eigen::VectorXd A0(4, 1);
        A0 << -(coeffs[3] * cc / D(k).real() + coeffs[4] * cc + coeffs[6] / D(k).real() + coeffs[7]),
            -(coeffs[11] * cc / D(k).real() + coeffs[12] * cc + coeffs[14] / D(k).real() + coeffs[15]),
            -(coeffs[19] * cc / D(k).real() + coeffs[20] * cc + coeffs[22] / D(k).real() + coeffs[23]),
            -(coeffs[27] * cc / D(k).real() + coeffs[28] * cc + coeffs[30] / D(k).real() + coeffs[31]);

        Eigen::VectorXd xz = A1.partialPivLu().solve(A0);
        sols(1, m) = xz[3];                        // u
        sols(2, m) = xz[2];                        // v
        sols(0, m) = std::sqrt(V0(0, k).real());   // c x
        sols(3, m) = 1.0 / std::sqrt(D(k).real()); // f
        ++m;
    }

    sols.conservativeResize(4, m);
    return sols;
}

void shared_focal_abspose_single_perm(const std::vector<Eigen::Vector3d> &x1h, const std::vector<Eigen::Vector3d> &x2h,
                                      const std::vector<double> &depth1, const std::vector<double> &depth2,
                                      std::vector<ImagePair> *models) {
    Eigen::Vector3d X101 = depth1[0] * x1h[0] - depth1[1] * x1h[1];
    Eigen::Vector3d X201 = depth2[0] * x2h[0] - depth2[1] * x2h[1];

    double f2 = (X101(0) * X101(0) + X101(1) * X101(1) - X201(0) * X201(0) - X201(1) * X201(1)) /
                (X201(2) * X201(2) - X101(2) * X101(2));

    if (f2 > 0) {
        double f = sqrt(f2);

        // if (f > 4.0 || f < 0.2)
        //     f = 1.2;
        Eigen::DiagonalMatrix<double, 3> Kinv(1.0/f, 1.0/f, 1.0);

        Eigen::Vector3d X10 = depth1[0] * Kinv * x1h[0];
        Eigen::Vector3d X20 = depth2[0] * Kinv * x2h[0];

        Eigen::Vector3d X11 = depth1[1] * Kinv * x1h[1];
        Eigen::Vector3d X21 = depth2[1] * Kinv * x2h[1];

        const double c2 = X10.squaredNorm(), c3 = X11.squaredNorm(), c4 = X20.squaredNorm(), c5 = X21.squaredNorm(),
                     b0 = 2 * X20.transpose() * x2h[2], b1 = 2 * X10.transpose() * x1h[2],
                     b2 = 2 * X21.transpose() * x2h[2], b3 = 2 * X11.transpose() * x1h[2],

                     b4 = x1h[2].squaredNorm(), b5 = x2h[2].squaredNorm();

        const double a0 = b2 - b0, a1 = (b3 - b1) / a0, a2 = (c2 - c4 + c5 - c3) / a0;
        static const double TOL_IMAG = 1e-3;
        double lambda2[2];
        bool real2[2];
        univariate::solve_quadratic_real_tol(b5 * a1 * a1 - b4, b1 - a1 * b0 + 2 * a1 * a2 * b5,
                                             b5 * a2 * a2 - b0 * a2 - c2 + c4, lambda2, real2, TOL_IMAG);

        for (int m = 0; m < 2; ++m) {
            if (!real2[m])
                continue;
            if (lambda2[m] < 0)
                continue;

            double lambda2s = a1 * lambda2[m] + a2;
            if (lambda2s < 0)
                continue;

            Eigen::Vector3d v1 = X20 - X21;
            Eigen::Vector3d v2 = X20 - lambda2s * x2h[2];
            Eigen::Matrix3d Y;
            Y << v1, v2, v1.cross(v2);

            Eigen::Vector3d u1 = X10 - X11;
            Eigen::Vector3d u2 = X10 - lambda2[m] * x1h[2];
            Eigen::Matrix3d X;
            X << u1, u2, u1.cross(u2);
            X = X.inverse().eval();

            Eigen::Matrix3d rot = Y * X;
            // if rot is less than 5 deg
            //            if ((rot.trace() - 1) > 1.99238939618)
            //                if (f < opt.min_focal_1 or f > opt.max_focal_1)
            //                    continue;

            Eigen::Vector3d trans = X20 - rot * X10;

            CameraPose pose = CameraPose(rot, trans);
            Camera camera = Camera("SIMPLE_PINHOLE", std::vector<double>{f, 0.0, 0.0}, -1, -1);
            models->emplace_back(pose, camera, camera);
        }
    }
}


void shared_focal_monodepth_3p(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                                    const std::vector<Eigen::Vector2d> &sigma,
                                    std::vector<ImagePair> *models, const RansacOptions &opt) {
    models->clear();
    if (opt.all_permutations)
        models->reserve(6);
    else
        models->reserve(2);

    std::vector<Eigen::Vector3d> x1h(3);
    std::vector<Eigen::Vector3d> x2h(3);
    for (int i = 0; i < 3; ++i) {
        x1h[i] = x1[i].homogeneous();
        x2h[i] = x2[i].homogeneous();
    }

    std::vector<double> depth1(3);
    std::vector<double> depth2(3);
    for (int i = 0; i < 3; ++i) {
        depth1[i] = sigma[i][0];
        depth2[i] = sigma[i][1];
    }

//    Eigen::VectorXd datain(18);
//    datain << x1h[0][0], x1h[1][0], x1h[2][0], x1h[0][1], x1h[1][1], x1h[2][1], x2h[0][0], x2h[1][0], x2h[2][0],
//        x2h[0][1], x2h[1][1], x2h[2][1], depth1[0], depth1[1], depth1[2], depth2[0], depth2[1], depth2[2];

    shared_focal_abspose_single_perm(x1h, x2h, depth1, depth2, models);

    if (opt.all_permutations){
        std::swap(x1h[1], x1h[2]);
        std::swap(x2h[1], x2h[2]);
        std::swap(depth1[1], depth1[2]);
        std::swap(depth2[1], depth2[2]);
        shared_focal_abspose_single_perm(x1h, x2h, depth1, depth2, models);

        std::swap(x1h[0], x1h[2]);
        std::swap(x2h[0], x2h[2]);
        std::swap(depth1[0], depth1[2]);
        std::swap(depth2[0], depth2[2]);
        shared_focal_abspose_single_perm(x1h, x2h, depth1, depth2, models);
    }
}


void shared_focal_monodepth_4p(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                                    const std::vector<Eigen::Vector2d> &sigma, bool use_eigen,
                                    std::vector<ImagePair> *models) {
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
    if (use_eigen) {
        sols = shared_focal_monodepth_relpose_eigen_impl(datain);
    } else {
        sols = shared_focal_monodepth_relpose_gb_impl(datain);
    }

    for (int k = 0; k < sols.cols(); ++k) {
        double s = sols(0, k);
        double u = sols(1, k);
        double v = sols(2, k);
        double f = sols(3, k);

        Eigen::Matrix3d Kinv;
        Kinv << 1.0 / f, 0, 0, 0, 1.0 / f, 0, 0, 0, 1;

        Eigen::Vector3d v1 = s * (depth2[0] + v) * Kinv * x2h[0] - s * (depth2[1] + v) * Kinv * x2h[1];
        Eigen::Vector3d v2 = s * (depth2[0] + v) * Kinv * x2h[0] - s * (depth2[2] + v) * Kinv * x2h[2];
        Eigen::Matrix3d Y;
        Y << v1, v2, v1.cross(v2);

        Eigen::Vector3d u1 = (depth1[0] + u) * Kinv * x1h[0] - (depth1[1] + u) * Kinv * x1h[1];
        Eigen::Vector3d u2 = (depth1[0] + u) * Kinv * x1h[0] - (depth1[2] + u) * Kinv * x1h[2];
        Eigen::Matrix3d X;
        X << u1, u2, u1.cross(u2);
        X = X.inverse().eval();

        Eigen::Matrix3d rot = Y * X;

        Eigen::Vector3d trans1 = (depth1[0] + u) * rot * Kinv * x1h[0];
        Eigen::Vector3d trans2 = s * (depth2[0] + v) * Kinv * x2h[0];
        Eigen::Vector3d trans = trans2 - trans1;

        CameraPose pose = CameraPose(rot, trans);
        pose.shift = u;
        Camera camera1 = Camera("SIMPLE_PINHOLE", std::vector<double>{f, 0.0, 0.0}, -1, -1);
        Camera camera2 = Camera("SIMPLE_PINHOLE", std::vector<double>{f, 0.0, 0.0}, -1, -1);
        models->emplace_back(pose, camera1, camera2);
    }
}
} //namespace poselib

