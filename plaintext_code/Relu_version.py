import torch
from torch.nn import Module
import torch.nn.functional as F


# import numpy as np

def square(r):
    return r * r


def tri(r):
    return r * r * r


def fou(r):
    return r * r * r * r


def sigmoid(x):
    # return 1/(1+np.exp(-x))
    return F.sigmoid(x)


#####################################  RELU DERI #####################################
# [6.25000002e-05 3.75000007e-01 4.99925000e-01 1.20000000e-01]
def relu_tri_deri_fit_1(x):
    return 6.25000002e-05 * (tri(x)) + 3.75000007e-01 * (square(x)) + 4.99925000e-01 * x + 1.20000000e-01


def relu_tri_deri_fit_2(x):
    return 7.81250049e-05 * (tri(x)) + 1.87500082e-01 * (square(x)) + 4.99625000e-01 * x + 0.3


def relu_tri_deri_fit_3(x):
    return 2.31481488e-05 * (tri(x)) + 1.25000024e-01 * (square(x)) + 4.99750000e-01 * x + 0.35


def relu_tri_deri_fit_4(x):
    return 9.76562515e-06 * tri(x) + 9.37500103e-02 * square(x) + 4.99812500e-01 * x + 0.5


def relu_tri_deri_fit_5(x):
    return 5.00000000e-07 * tri(x) + 7.50000001e-02 * square(x) + 4.99985000e-01 * x + 0.6


def relu_tri_deri_fit_6(x):
    return 2.89352053e-05 * (tri(x)) + 6.25003038e-02 * (square(x)) + 4.98750001e-01 * x + 0.7


def relu_tri_deri_fit_7(x):
    return 1.82215743e-07 * (tri(x)) + 5.35714286e-02 * (square(x)) + 4.99989286e-01 * x + 0.85


def relu_tri_deri_fit_8(x):
    return 1.22070360e-05 * (tri(x)) + 4.68751282e-02 * (square(x)) + 4.99062500e-01 * x + 0.95


def relu_four_deri_fit_2(x):
    return -3.41796872e-02 * fou(x) + 9.76562534e-07 * tri(x) + 3.51562499e-01 * square(
        x) + 4.99978906e-01 * x + 1.40000000e-01


def relu_four_deri_fit_3(x):
    return -1.01273148e-02 * fou(x) + -2.02546271e-06 * tri(x) + 2.34375000e-01 * square(
        x) + 5.00023437e-01 * x + 2.20000000e-01


def relu_four_deri_fit_4(x):
    return -4.27246094e-03 * fou(x) + -8.54492045e-07 * tri(x) + 1.75781250e-01 * square(
        x) + 5.00017578e-01 * x + 2.90000000e-01


def relu_four_deri_fit_5(x):
    return -2.18750000e-03 * fou(x) + 6.24998987e-08 * tri(x) + 1.40625000e-01 * square(
        x) + 4.99991563e-01 * x + 3.50000000e-01


def relu_four_deri_fit_6(x):
    return -1.26591435e-03 * fou(x) + 3.61689109e-08 * tri(x) + 1.17187500e-01 * square(
        x) + 4.99992969e-01 * x + 4.30000000e-01


def relu_four_deri_fit_7(x):
    return -7.97193877e-04 * fou(x) + 2.27769160e-08 * tri(x) + 1.00446429e-01 * square(
        x) + 4.99993973e-01 * x + 4.90000000e-01


def relu_four_deri_fit_8(x):
    return -5.34057617e-04 * fou(x) + 1.52587493e-08 * tri(x) + 8.78906250e-02 * square(
        x) + 4.99994727e-01 * x + 5.50000000e-01


#####################################  RELU Direct Poly #####################################
def relu_tri_fit_1(x):
    return 5.46875015e-05 * (tri(x)) + 4.68750009e-01 * (square(x)) + 4.99976562e-01 * x + 9.37499977e-02


def relu_tri_fit_2(x):
    return 6.83593757e-06 * (tri(x)) + 2.34375001e-01 * (square(x)) + 4.99988281e-01 * x + 1.87499999e-01


def relu_tri_fit_3(x):
    return 2.02546271e-06 * (tri(x)) + 1.56250000e-01 * (square(x)) + 4.99992188e-01 * x + 2.81249999e-01


def relu_tri_fit_4(x):
    return 8.54492044e-07 * tri(x) + 1.17187500e-01 * square(x) + 4.99994141e-01 * x + 3.74999999e-01


def relu_tri_fit_5(x):
    return 4.37500102e-07 * tri(x) + 9.37500001e-02 * square(x) + 4.99995312e-01 * x + 4.68750000e-01


def relu_tri_fit_6(x):
    return 2.53182941e-07 * (tri(x)) + 7.81250000e-02 * (square(x)) + 4.99996094e-01 * x + 5.62500000e-01


def relu_tri_fit_7(x):
    return 1.59438828e-07 * (tri(x)) + 6.69642857e-02 * (square(x)) + 4.99996652e-01 * x + 6.56250000e-01


def relu_tri_fit_8(x):
    return 1.06811563e-07 * (tri(x)) + 5.85937500e-02 * (square(x)) + 4.99997070e-01 * x + 7.50000000e-01


def relu_tri_fit_9(x):
    return 1.851e-04 * (tri(x)) + 0.250 * (square(x)) + 4.99e-01 * x + 1.80e-01


def relu_four_fit_2(x):
    return -5.12695316e-02 * fou(x) + -3.41796879e-06 * tri(x) + 4.10156251e-01 * square(
        x) + 5.00005859e-01 * x + 1.17187499e-01


def relu_four_fit_3(x):
    return -1.51909723e-02 * fou(x) + -1.01273136e-06 * tri(x) + 2.73437500e-01 * square(
        x) + 5.00003906e-01 * x + 1.75781250e-01


def relu_four_fit_4(x):
    return -6.40869142e-03 * fou(x) + -4.27246022e-07 * tri(x) + 2.05078125e-01 * square(
        x) + 5.00002930e-01 * x + 2.34375000e-01


def relu_four_fit_5(x):
    return -3.28125000e-03 * fou(x) + -2.18750051e-07 * tri(x) + 1.64062500e-01 * square(
        x) + 5.00002344e-01 * x + 2.92968750e-01


def relu_four_fit_6(x):
    return -1.89887153e-03 * fou(x) + -1.26591471e-07 * tri(x) + 1.36718750e-01 * square(
        x) + 5.00001953e-01 * x + 3.51562500e-01


def relu_four_fit_7(x):
    return -1.19579082e-03 * fou(x) + -7.97194138e-08 * tri(x) + 1.17187500e-01 * square(
        x) + 5.00001674e-01 * x + 4.10156250e-01


def relu_four_fit_8(x):
    return -8.01086426e-04 * fou(x) + -5.34057816e-08 * tri(x) + 1.02539063e-01 * square(
        x) + 5.00001465e-01 * x + 4.68750000e-01


#####################################  SWISH  DERI #####################################
def swish_tri_deri_fit_1(x):
    return 3.39715908e-06 * tri(x) + 2.27423893e-01 * square(x) + 4.99997962e-01 * x + 3.40000000e-03


def swish_tri_deri_fit_2(x):
    return 2.02821689e-06 * tri(x) + 1.80147532e-01 * square(x) + 4.99995132e-01 * x + 7.00000000e-02


def swish_tri_deri_fit_3(x):
    return 1.05922684e-06 * tri(x) + 1.36149522e-01 * square(x) + 4.99994280e-01 * x + 1.00000000e-01


def swish_tri_deri_fit_4(x):
    return 5.47965325e-07 * tri(x) + 1.04152864e-01 * square(x) + 4.99994740e-01 * x + 3.00000000e-01


def swish_tri_deri_fit_5(x):
    return 2.97233723e-07 * tri(x) + 8.23781087e-02 * square(x) + 4.99995541e-01 * x + 4.00000000e-01


def swish_tri_deri_fit_6(x):
    return 1.72077791e-07 * tri(x) + 6.74730644e-02 * square(x) + 4.99996283e-01 * x + 5.00000000e-01


def swish_tri_deri_fit_7(x):
    return 1.06310176e-07 * tri(x) + 5.69411090e-02 * square(x) + 4.99996874e-01 * x + 6.50000000e-01


def swish_tri_deri_fit_8(x):
    return 6.95945181e-08 * tri(x) + 4.92128494e-02 * square(x) + 4.99997328e-01 * x + 8.00000000e-01


def swish(x):
    return x * sigmoid(x)


def swish_four_deri_fit_2(x):
    return -1.17922217e-02 * fou(x) + -3.30227456e-07 * tri(x) + 2.36750196e-01 * square(
        x) + 5.00000793e-01 * x + 1.00000000e-04


def swish_four_deri_fit_3(x):
    return -6.94293543e-03 * fou(x) + -3.29360065e-07 * tri(x) + 2.11133225e-01 * square(
        x) + 5.00001779e-01 * x + 1.00000000e-02


def swish_four_deri_fit_4(x):
    return -3.98819483e-03 * fou(x) + -2.49673506e-07 * tri(x) + 1.80726204e-01 * square(
        x) + 5.00002397e-01 * x + 5.00000000e-02


def swish_four_deri_fit_5(x):
    return -2.34401466e-03 * fou(x) + -1.71569318e-07 * tri(x) + 1.52698548e-01 * square(
        x) + 5.00002574e-01 * x + 1.00000000e-01


def swish_four_deri_fit_6(x):
    return -1.43568924e-03 * fou(x) + -1.15060137e-07 * tri(x) + 1.29494839e-01 * square(
        x) + 5.00002485e-01 * x + 2.00000000e-01


def swish_four_deri_fit_7(x):
    return -9.20252917e-04 * fou(x) + -7.77404671e-08 * tri(x) + 1.11051980e-01 * square(
        x) + 5.00002286e-01 * x + 2.50000000e-01


def swish_four_deri_fit_8(x):
    return -6.16108480e-04 * fou(x) + -5.36272239e-08 * tri(x) + 9.65299807e-02 * square(
        x) + 5.00002059e-01 * x + 3.40000000e-01


#####################################  SWISH Direct Poly #####################################
def swish_tri_fit_1(x):
    return 3.55225479e-06 * (tri(x)) + 2.33513472e-01 * (square(x)) + 4.99998478e-01 * x + 1.60482648e-03


def swish_tri_fit_2(x):
    return 2.35844435e-06 * (tri(x)) + 1.96319722e-01 * (square(x)) + 4.99995957e-01 * x + 1.94411072e-02


def swish_tri_fit_3(x):
    return 1.38858691e-06 * (tri(x)) + 1.57573437e-01 * (square(x)) + 4.99994644e-01 * x + 6.81049352e-02


def swish_tri_fit_4(x):
    return 7.97638832e-07 * tri(x) + 1.26030961e-01 * square(x) + 4.99994530e-01 * x + 1.44926156e-01


def swish_tri_fit_5(x):
    return 4.68803042e-07 * tri(x) + 1.02469663e-01 * square(x) + 4.99994977e-01 * x + 2.39653410e-01


def swish_tri_fit_6(x):
    return 2.87137928e-07 * (tri(x)) + 8.51935716e-02 * (square(x)) + 4.99995570e-01 * x + 3.43487857e-01


def swish_tri_fit_7(x):
    return 1.84050644e-07 * (tri(x)) + 7.24013580e-02 * (square(x)) + 4.99996135e-01 * x + 4.50990901e-01


def swish_tri_fit_8(x):
    return 1.23221742e-07 * (tri(x)) + 6.27320298e-02 * (square(x)) + 4.99996620e-01 * x + 5.59285654e-01


def swish_four_fit_2(x):
    return -1.29842810e-02 * fou(x) + -2.38411855e-07 * tri(x) + 2.40837257e-01 * square(
        x) + 5.00000409e-01 * x + 1.63409355e-03


def swish_four_fit_3(x):
    return -8.26021383e-03 * fou(x) + -2.63455646e-07 * tri(x) + 2.21295087e-01 * square(
        x) + 5.00001016e-01 * x + 1.07554509e-02


def swish_four_fit_4(x):
    return -5.08479681e-03 * fou(x) + -2.19320358e-07 * tri(x) + 1.95765317e-01 * square(
        x) + 5.00001504e-01 * x + 3.33511868e-02


def swish_four_fit_5(x):
    return -3.15851783e-03 * fou(x) + -1.62900671e-07 * tri(x) + 1.70152188e-01 * square(
        x) + 5.00001745e-01 * x + 7.04470985e-02


def swish_four_fit_6(x):
    return -2.01670236e-03 * fou(x) + -1.16202656e-07 * tri(x) + 1.47423244e-01 * square(
        x) + 5.00001793e-01 * x + 1.19461036e-01


def swish_four_fit_7(x):
    return -1.33196747e-03 * fou(x) + -8.23429384e-08 * tri(x) + 1.28343992e-01 * square(
        x) + 5.00001729e-01 * x + 1.76871995e-01


def swish_four_fit_8(x):
    return -9.10509563e-04 * fou(x) + -5.88802384e-08 * tri(x) + 1.12679983e-01 * square(
        x) + 5.00001615e-01 * x + 2.39618754e-01


#####################################  MISH DERI #####################################
def mish_tri_deri_fit_1(x):
    return -0.0155181 * tri(x) + 0.27354573 * square(x) + 0.59977063 * x + 0.01023


def mish_tri_deri_fit_2(x):
    return -0.0109158 * tri(x) + 0.19442432 * square(x) + 0.59278305 * x + 0.1


def mish_tri_deri_fit_3(x):
    return -0.00571553 * tri(x) + 0.13727469 * square(x) + 0.57347461 * x + 0.2


def mish_tri_deri_fit_4(x):
    return -0.00269221 * tri(x) + 0.10231588 * square(x) + 0.55182959 * x + 0.3


def mish_tri_deri_fit_5(x):
    return -0.00126304 * tri(x) + 0.08048166 * square(x) + 0.5348917 * x + 0.5


def mish_tri_deri_fit_6(x):
    return -0.00061492 * tri(x) + 0.06603264 * square(x) + 0.52337135 * x + 0.6


def mish_tri_deri_fit_7(x):
    return -3.15211603e-04 * tri(x) + 5.59161188e-02 * square(x) + 5.15902272e-01 * x + 7.00000000e-01


def mish_tri_deri_fit_8(x):
    return -1.70647523e-04 * tri(x) + 4.84861093e-02 * square(x) + 5.11090748e-01 * x + 8.50000000e-01


def mish(x):
    return x * F.tanh(torch.log(1 + torch.exp(x)))


def mish_four_deri_fit_2(x):
    return -0.01858827 * fou(x) + -0.01091952 * tri(x) + 0.283648 * square(x) + 0.59279197 * x + 0.001


def mish_four_deri_fit_3(x):
    return -0.00893021 * fou(x) + -0.00571732 * tri(x) + 0.23372091 * square(x) + 0.57348426 * x + 0.05


def mish_four_deri_fit_4(x):
    return -0.00449032 * fou(x) + -0.00269311 * tri(x) + 0.18853003 * square(x) + 0.55183821 * x + 0.1


def mish_four_deri_fit_5(x):
    return -0.00244631 * fou(x) + -0.00126353 * tri(x) + 0.15387088 * square(x) + 0.53489904 * x + 0.2


def mish_four_deri_fit_6(x):
    return -0.00143936 * fou(x) + -0.00061521 * tri(x) + 0.12821286 * square(x) + 0.52337756 * x + 0.2


def mish_four_deri_fit_7(x):
    return -9.04394869e-04 * fou(x) + -3.15392482e-04 * tri(x) + 1.09094537e-01 * square(
        x) + 5.15907590e-01 * x + 3.00000000e-01


def mish_four_deri_fit_8(x):
    return -6.00097239e-04 * fou(x) + -1.70767543e-04 * tri(x) + 9.45735772e-02 * square(
        x) + 5.11095357e-01 * x + 4.00000000e-01


#####################################  MISH Direct Poly #####################################
def mish_tri_fit_1(x):
    return -0.01573175 * (tri(x)) + 0.28558935 * (square(x)) + 0.59990003 * x + 0.00328814


def mish_tri_fit_2(x):
    return -0.01229217 * (tri(x)) + 0.21991659 * (square(x)) + 0.59608887 * x + 0.03404072


def mish_tri_fit_3(x):
    return -0.00753087 * (tri(x)) + 0.16483048 * (square(x)) + 0.58328021 * x + 0.10230162


def mish_tri_fit_4(x):
    return -0.00415123 * tri(x) + 0.12694828 * square(x) + 0.56583866 * x + 0.19398058


def mish_tri_fit_5(x):
    return -0.00224187 * tri(x) + 0.10144986 * square(x) + 0.54957629 * x + 0.29620854


def mish_tri_fit_6(x):
    return -0.00123175 * (tri(x)) + 0.08379832 * (square(x)) + 0.53669648 * x + 0.40217083


def mish_tri_fit_7(x):
    return -0.00069875 * (tri(x)) + 0.0711099 * (square(x)) + 0.52717971 * x + 0.50874746


def mish_tri_fit_8(x):
    return -4.11144186e-04 * (tri(x)) + 6.16539212e-02 * (square(x)) + 5.20327137e-01 * x + 6.14631827e-01


def mish_four_fit_2(x):
    return -0.0215476 * fou(x) + -0.01229648 * tri(x) + 0.29379408 * square(x) + 0.59609626 * x + 0.00448972


def mish_four_fit_3(x):
    return -0.01136236 * fou(x) + -0.00753314 * tri(x) + 0.25248299 * square(x) + 0.58328898 * x + 0.02341436


def mish_four_fit_4(x):
    return -0.00610427 * fou(x) + -0.00415246 * tri(x) + 0.21066405 * square(x) + 0.56584703 * x + 0.06003535


def mish_four_fit_5(x):
    return -0.00347365 * fou(x) + -0.00224257 * tri(x) + 0.17588525 * square(x) + 0.54958374 * x + 0.11012008


def mish_four_fit_6(x):
    return -0.00210237 * fou(x) + -0.00123217 * tri(x) + 0.14867138 * square(x) + 0.53670297 * x + 0.16862782


def mish_four_fit_7(x):
    return -0.00134521 * fou(x) + -0.00069902 * tri(x) + 0.12760862 * square(x) + 0.52718536 * x + 0.23190373


def mish_four_fit_8(x):
    return -9.02898167e-04 * fou(x) + -4.11324766e-04 * tri(x) + 1.11184335e-01 * square(
        x) + 5.20332090e-01 * x + 2.97637179e-01


###################################################  tanh deri #######################################################
def tanh_tri_deri_fit_2(x):
    return -8.17529195e-02 * tri(x) + -4.54988235e-06 * square(x) + 8.09025468e-01 * x + 0.0


def tanh_tri_deri_fit_3(x):
    return -3.43292558e-02 * tri(x) + -2.46756404e-06 * square(x) + 6.40648220e-01 * x + 0.0


def tanh_tri_deri_fit_4(x):
    return -1.65565398e-02 * tri(x) + -1.31867742e-06 * square(x) + 5.14736961e-01 * x + 0.0


def tanh_tri_deri_fit_5(x):
    return -9.01545474e-03 * tri(x) + -7.52917614e-07 * square(x) + 4.25368209e-01 * x + 0.0


###################################################  tanh Direct #######################################################

def tanh_tri_fit_2(x):
    return -1.05675323e-01 * (tri(x)) + -3.58836046e-06 * (square(x)) + 8.66439236e-01 * x + 2.87068836e-06


def tanh_tri_fit_3(x):
    return -4.94735597e-02 * (tri(x)) + -2.27164530e-06 * (square(x)) + 7.22427461e-01 * x + 4.08896154e-06


def tanh_tri_fit_4(x):
    return -2.54882785e-02 * tri(x) + -1.33976059e-06 * square(x) + 6.00481653e-01 * x + 4.28723388e-06


def tanh_tri_fit_5(x):
    return -1.44411486e-02 * tri(x) + -8.13854272e-07 * square(x) + 5.06753618e-01 * x + 4.06927136e-06


###################################################  sigmoid deri #######################################################
def sigmoid_tri_deri_fit_2(x):
    return -1.28137584e-02 * tri(x) + -3.20720889e-07 * square(x) + 2.41653573e-01 * x + 0.5


def sigmoid_tri_deri_fit_3(x):
    return -8.17140003e-03 * tri(x) + -3.45031773e-07 * square(x) + 2.24400643e-01 * x + 0.5


def sigmoid_tri_deri_fit_4(x):
    return -5.10955747e-03 * tri(x) + -2.84367598e-07 * square(x) + 2.02256367e-01 * x + 0.5


def sigmoid_tri_deri_fit_5(x):
    return -3.25666423e-03 * tri(x) + -2.12459565e-07 * square(x) + 1.80078036e-01 * x + 0.5


###################################################  sigmoid Direct #######################################################

def sigmoid_tri_fit_2(x):
    return -1.41758075e-02 * (tri(x)) + -2.04307356e-07 * (square(x)) + 2.44922491e-01 * x + 5.00000163e-01


def sigmoid_tri_fit_3(x):
    return -9.80935285e-03 * (tri(x)) + -2.45692890e-07 * (square(x)) + 2.33245588e-01 * x + 5.00000442e-01


def sigmoid_tri_fit_4(x):
    return -6.60470766e-03 * tri(x) + -2.24272491e-07 * square(x) + 2.16609809e-01 * x + 5.00000718e-01


def sigmoid_tri_fit_5(x):
    return -4.47353033e-03 * tri(x) + -1.82529957e-07 * square(x) + 1.98331027e-01 * x + 5.00000913e-01


########################################################################################################################
def relu_my_1(x):
    return 2.025e-06 * tri(x) + 0.156 * square(x) + 0.499 * x + 0.281

def relu_my_2(x):
    return 6.250e-05 * tri(x) + 0.375 * square(x) + 4.999 * x + 0.125

def relu_my_3(x):
    return 6.835e-06 * tri(x) + 0.234 * square(x) + 0.499 * x + 0.187

def relu_my_4(x):
    return 1.851e-04 * tri(x) + 0.250 * square(x) + 0.499 * x + 0.180

def relu_my_5(x):
    return square(x)

def relu_my_6(x):
    return 2.0e-03 * square(x) + 2.0e-01 * x + 2.0e-02


def poly_relu(x):

    return relu_tri_fit_2(x) ########mark
    



class PolyReLU(Module):

    def forward(self, x):
        return poly_relu(x)