from collections import namedtuple

mb_convblock = namedtuple("MBConv", ["expand_ratio", "repeats", "kernel_size", "channels", "stride"])
compound_val = namedtuple("CompoundVal", ["phi_val", "resolution", "drop_rate"])

base_model = [
    mb_convblock(1, 1, 3, 16, 1),
    mb_convblock(6, 2, 3, 24, 2),
    mb_convblock(6, 2, 5, 40, 2),
    mb_convblock(6, 3, 3, 80, 2),
    mb_convblock(6, 3, 5, 112, 1),
    mb_convblock(6, 4, 5, 192, 2),
    mb_convblock(6, 1, 3, 320, 1)
]

compound_params_dict = {
    "b0": compound_val(0, 224, 0.2),
    "b1": compound_val(0.5, 240, 0.2),
    "b2": compound_val(1, 260, 0.3),
    "b3": compound_val(2, 300, 0.3),
    "b4": compound_val(3, 380, 0.4),
    "b5": compound_val(4, 456, 0.4),
    "b6": compound_val(5, 528, 0.5),
    "b7": compound_val(6, 600, 0.5)
}