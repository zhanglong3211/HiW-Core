# -*- coding: utf-8 -*-
"""Run all 12 SROT experiment result packages from one Python entry point.

Put this file in the same directory as these dataset archives, then run:

    python run_all_12_experiments.py

Required external dataset archives:
    - cifar100_10classes_500each_5000.zip
    - facescrub_top10_2k.zip

The Digits dataset is loaded from scikit-learn. By default the script runs the
same 50 repeats and 25 budget points used by the original experiment scripts.
Use --data-dir and --out-dir when the dataset zips and outputs are not in the
same directory as this file.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import py_compile
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

CIFAR_ZIP_NAME = 'cifar100_10classes_500each_5000.zip'
FACESCRUB_ZIP_NAME = 'facescrub_top10_2k.zip'
DEFAULT_BUDGET_RATIOS = [round(x / 100.0, 2) for x in range(2, 51, 2)]

SCRIPT_PAYLOADS = {
    'cifar_kcenter': (
        'H4sIAL6IJ2oC/7U9TXPjuHJ3/wokWy8kPRQtacazs/Jq63nGnt2J58M1dpKNFYVFiZBEWyJpkrKtcfldc8o551Tl'
        'mEv+QvJbkqr8i3Q3ABIgKdtvX97szooEGo1Go9HobjS4syxZMd+frYt1xn2fRas0yQoWxHFSBEWUxPmOLEpyl62C'
        'YuGyIlpxl32L0lm0hId8sS6ipcsu8yR22W2QxVE8z3dmiDmFBstootCeUvtT6Oo0yaM7fBVwUaJA3m4Knn/4spPk'
        'Ho9voiyJvZwXIZ8F62VhW19Ojz+//Xh45n/+m0/++S9fjw+PzizX6lnO1hafTp8P/OnkYwuwJC1er9INC3IWp6oo'
        'DeIQCuDfNFRlwKR0mRQw7J3q0Vvn3LYO5/MKnVaZbvCJ0CwLwZHTDx8VSz6sgjnwmX6+pJKz+dWSA6u9VRLypZ/z'
        'JZ/idKk2RRZEsQ+sLPw8XUaF2SjNeJolU57nMFWqyVmBg8nCs2mw5JnZIOTTBKDySO/j9N2hgoqQNlU+TZZJ5iIJ'
        'cT5LspUB5M14gLKmgBfJXNQXi4wHYZoky2mxLIdRFvrLaBUV+U6jxBY/w56zo2TPA8EseKZebSuax0nGgfU7R4fn'
        'h/7Fh1M2JGG0rb1VXOyFQRHsTaNZkPW6Xb/XnS6DPOe5v9/t8mC6wN+uBwIPGL5++XL+xzf28ywp/Jvcv5ryGEjz'
        'ozgq/FsA9C9zH2Tt434XkL//8LN/9OEr4Kdu9pg1i+bArNza+Xp8VquCYpBZqAIes5RFMRthlcskFpfJNuPBDoM/'
        'qbe6CqPMToMMaMiH59kapIrfRSAjyRW9AoM+AzWnx4fnZ9DTfnfn7eHZsX92fHwEr/1u//XO2785+vn43P96eP7h'
        'C8Isobkdp16WrOMQHwKY9jm3u16377Kut9/D/3b7jsv6jrPz4fOHc//Tl6NjbDuyVjyILRfGAqKXrPAJ2FgsUGxR'
        'KHxVf4UPuS+YhwWr4M6HOV1Z453Tj1/O/S9fj46/EsrTLAFB5WHn7OuXc2vMXrDRzLrqvKOmDPk+vL97sBhy7Q65'
        'VpE03tn5BG9vD8/f/eKffDo+/HzmQ4EvCs4+XBxDD73+myaUeHl/+O78C1LxsglBvZQA/SbA2fnx6Zl/evzV/3BO'
        'Q+nt7IA28LLpKbB0lXvrFCSN2/c0mVIwvDCNrAHrvekCT/LghkOxLHu1j2WzJC68WbCKlhsoA5A47+Q8i2aWK/Eg'
        'gFY8ABYeZlGwRC7/wpc3vIimAb58jCY8oz2BnQE8Fh3xy+Bv1+J1bCCMvnFA9YO3D1DBHc+9ZTDhS1nc61blRVQs'
        'uSrveUjzks95HHqISFa8gXKBXeCKYn4bhcUC21ATKubhnJPuwZF+16c/lqqcBdOq8nYRFZykTXCxrVL0h12ZHfa9'
        'LhIvKlZBdsUzSeU+kXIH/LryYKEJhYwYI5LhTXuF6Ee0WgWXSaaY99J7U7ZqqZAtoriq6Huvqhb1Cr2fNi5tttYQ'
        '/+ZZFEL5+2CZc22OQDI5jUVUiE7ScEazV2xS7P0VKAIrzWtFOw/OzrsvH798RU1wX1u2A+u714cvj344hB3YXLuk'
        'EaC62/3h+PuXjWqpSAjg+/7bfgOgRb8A8NH+/nG32wA21Q7AHb/+4X0LXKmNkG76Yz3sfDr8enK8ZXSn28Z1tH1E'
        '//i8sdw8OYr8Mfp/Bco/fvh8fHb+9x+P24nvbCN+sJ34Tud51He8J8lvQaXRPwD6dz58+tk//pX2MNvyLtM5ijH8'
        'cvGQxuJ3skrp95ZPcGvf2QE7kOXXIW5phy5764iN8xDQ4MaWB1kWbLAmRDEeQtlsmQTF61fOAXtrAr1tASJkRwIu'
        'X6/sw11AFcD+C7bLaOCyz0nMcb+S1W9331bVWOeywZh1UAGxXWYfst+zt965wJpxsKlibAmsiFbQ+oi2XTWoWQB8'
        'Foy0L8COB6ud83AI+ir2Qdtlw15XDjYDg5AoFFPnSdPYh3Ib28BQL8yhXrTyA1yBEOAuvHwRpPyAreAFdJIdxYW9'
        'AmMgFoRHM6j5cci6oneiIIhyzv42WK75cZYlmW2t2GqdF2zCmTBBb9CUK1v/BNRorUtOSEMk1siDzoE4F6iaJunG'
        'FkgmQQFWGmpKSWKMjs6d/YQp4LLHrYBdtnIklWjrNfFX/bag0s2FClPs5wVPc8QDCEDZIzfFDDoA9oRNoVEThXeA'
        'BObUmy6SaEpcQkqGJa0ucDJdwr44JM2O0451OKUjhWJMCMXCzIVQ8FVabGwbBCx0WuVCQo+6Y0SFmEY4CCQFfmFX'
        'yXK7JMJxRBfANT/sVyvHFrR0NGww/l3WLxcMNUMT7xJNPCEJYImunEpSCvBwl4CTaLNFF4RdMpuYRXx6jECUQYEJ'
        'pLjHO70+AzuHl20kfzW+pkM5nD3RUOtODedSY45ic50P8EwLXZS5W1lzWWeN6G4K9nohJ+0bz5Kc5qxVZSEb/YqN'
        'Ugg1Tk62ipMu43V5+nVCwjQxRhiDuwl7gtQw2RzXi9TIv05cNSjHnGdF5BSJhHbrOLpeAxECl0Yo8ZCvJkJcf52M'
        'yu6GbDoG4YyLUiCWPLYlrINyS/waTUFBDxHOwMmLgFEpzCmuzBLYRfvUMWDVvExxjm2ohrmC9rh+taoXhHNXUeuh'
        '6rZpzN1qH5GMuaj4csCEnW3w76jklhoh1E2imKi0RQMXZQuGPC8Ww5UDuh0lwSZOSNXDbzVhAd/7ituq11JKwmhV'
        'yUmoMR5a4w4H9eMt3d/yaL4Ar/RCwZkUCRrA9RBs/ol1FVUjKB2zPSrHZ7mPqv0hTgpixXJpQ6VG0jKZ0vI/Gml7'
        'BQiWIAiEAeyTko15khV2h5oAE9dgDCH7eWG3C+Dtgmfc/gN2CKrJlD+Ew/UCkNCDWSdpRt0BIA4RD3DYXxNQjZ8E'
        '6WKkGo0FfV4QhrYqA5onGQ+u6vqa31JJi8w0RctccnKbFS2VuGg7rGwlzQ+fQj9+DM5CLsy9b1Fqf5sNVBTRu4jS'
        '9/CrjBCBHbnOQ3sUE9NiZMW3mYdYKOpAujf2lsktz2zHA38kv42Kha2sP4eBCSMEoKq09ixnrMwin2I1JUU+hitt'
        'xD9geQFbaucn/JXhEzBXkUdG9JKAHY/qCOo79ndJdpUTwRz6AwuV+tiLVnO0OxmUZ0lS7BmlnpJWVDqEzGE/sv4j'
        'NtHMekcxWuDJTPUBnS5DeKFw2j1S9mAZs0WYR53+WA4fI2l+GblCBsjgHOxwNGGkw3uv5azI1aRCaB6FjnLbqZOJ'
        'E/k5Kd5jREhQq5oIanAe6hNfgWAUFCSjUh0oNMD3bTKk7Z40mQpeCs99+xw7lUhRg4c6miIBCwfdn+mARWLRumJ9'
        '83i9wkAI6L+qQ+ehbP/eZRuXIt9IxggUmfhbGR5dKMcgupfybOaTHkQJNlQJ9IaIzQ6pK9haC5hIWImmTiC2UpDY'
        'S1KQIxlJBxZ5ONFCVB3ib7Rq0TtooasYM87tzKcILrp+IA4OmMzxDQc1aH39+a3lNNqDM2D6BVHDoHjZd2CD7O/v'
        'e12j+XfsK4WAMNybrOeLMqBMOjWJc5AzHk83Yohpxm+iZJ2zfJpFaZF7Bi40RMrIMwwc8dpAkMs0mQbDuXxGxRYX'
        'kR8sowDD4TIuCr3kPLvhPu0MIjqq6Tk1HqPvLMA98g57RafH7vTM+jmwBTcwjLF42XzSxwL7zgRaAMQimdtY54K6'
        'iECTipOY4RsgK7oDheuD4IBDvFwOwQkQ6nYpSyewSV0NwdbtQTG9kGc8tD72O79sMGon17h/w6dFksm4r07Bey9I'
        'QYAomgvcn4LkxSh9IxifyxZjxwTfKHBt4Yy2alZUvnprWidlhwiwU98N2e8Y6KcuGmndptymGe5yMwsWbjBdwMb8'
        '7sP7w69qlDm7jx727lGvynWKK+q+ZfV1iu7A680eiEXLdb7QOPMepGMppPsGFt/0yn7fEAZlLW8qWLUSNg0XVNfK'
        'hNwV7Vy9GbEGJ7dSMypGgUc6Puxufinqleo28aHPDmIzDXwwqoavlJsvLPbK8kEGUQs5uQUe4HCxloyDJLQnsCoX'
        '7syw673sukwEC8AxADkZii5h4wSxnW2GAu3zgguyb+p3hP4EzNFqLeSfaIRdWfqEgjreDlV6ju9xIO8JlvgyKjJQ'
        'xvJRWokbhNkQzKaC2Rgw+RRqzRMyMDlmwA/owVFd4bYz9co5oboD2X2tgiuPng7WYi6cMTS95Fwh2ZmInoAN2enp'
        '7z25iAAULZJ3h7aOZ6i/gGWN8exYqrW2iboJ/TxZgm4fynMYmNdQavgLGhT0g2P16yO7oJFhbcvQ8ml/G88uFM8u'
        'JM/6GoILDXW9hhsrByBbVqErJhSAt1RyuYrmGYx/48+CaQSCvQF5XkXLIINH+wx05zqc86JyTNCrkIsnBq7GJBAq'
        'uiWAtRCXKMA417QMVUmgWrRLi1fhokdboU1ZQHtJCYtycnEqHKqClhbU5DZS2HDmYe+NhZWKEYotzZXykYWtMZyq'
        'gXraY6qBiKCA/0qxECkJdDpNPlMZIlLsrY0VPNR1pkcl4nYSgpsgWuoDniqwSZIsK4+0qPxROSXVuOeg2eSoVdT0'
        'DAMn66yMxoogqnfOfq+GarYe/YEIQQ+sQ2OYlfWXMnAkfargzqYW2uap2DIqsP2lHJQI/lCQRGOGIhAKXHaG9F2O'
        'jbWgkEnZRgWwLrh/vU6KILdPcBdZ5+TICQagtYeqsiyGzoK0dAJJFKneqTxbBVuKKIyvlM4T8Fg0s/K6NbSEDQ7Y'
        '9Uj3qrFbZzQ4GeO4ewdqQNc7Ghqa4xqWjK+AoRFtKScwb6udygar6nbFsPboRwjnAUZ4EhWxpGcbmpU7uiQSAz0C'
        'UPI5j8I1hQywN5zaaxkuPAAzDR0vM1qAOAFQoHAqkZTOPzUZDRRaLUpwTUE/7L4nls86S4HzgL7bFtbUY5owD9co'
        'QD/RcOHJtJcUJkBNYB0FdiDeh+pd+GkLcMzKNj/pWmuKnjU4N5dNYkoaflTIxjp56EFi64EWkVAYPWLcFd8Ml8Fq'
        'EgbscsBsicRV2DqI3MFwIuxYOa/ZryU51EfdlFRjIRVcI0Bw/lLw/aAE7ahZKMVSLLBUHoqJzA6aTDpaUecqK2PD'
        'kIvtQju2aGhxdULy6o04H+i9FvF9kVmBqVdefg38iTFU2PNe7oMz52jxG1eL7Gw57xEbfnXkI+JBclXjfJaxKxUS'
        'Ar2EIaz6NJNmkIEhaWnpaKZNnaGiG6BPQJJEE4qnSd6Rq1yKN8mX1rrikh73HCEYSIZ8gHV4VoZvj1BlwtoEGT8q'
        'G8OMAcDj+z4RCeicautXCqiswq20ejFDinI8pVMEICPo19TWalyu0aYhWiqrywee+Ce6FjcbSjG7Judyq+I/0Ddi'
        'jdmXLru6NAMN16ZOgeqf6r6XwuXxu4KTgKaoOU3CYD2NBleXymWUMqCaoiicDBr7IYVW5dYI20J7yx/1llpAtoQw'
        'VILQzzBEnbitcVmq/hMjsyV7pBxoodhmeLYVQ2PIsBhP2rTWk4B1G1PnbtPaFEKoEtWIF8KPP0JVcjRbo1tJ51mY'
        'e6g8zCtYbg1td/Cc42TUUnRS3DjjIIbLfnBQIl2s4vcsyuioSDOz8HhVHYR1ENzZ3dWOvhxp/jYwq0SFVtzG2V+8'
        'HUlbRsNjxIKC+k3EVukOz8N+sXvxDKy1PItKqz25uwDTe5hK4GjbzIve9+VO80YLrmrHjo8cQRn7C1imjZEJPFgH'
        'C0jiq+9cU9y5WjmEbUYtEoOGVz7WDk+njcPTsdPiQTXC8yVntztAuGBa3J9ybdIBORFdHYHjcRWuP7T/qWosExkO'
        'RP1IFNa8kZof1DMWa7urQtgcnRzho5gn0S0H0W5F4OW4JKtSOMWL3rhOX7sHc5n7YQQm3pzHU26nLrt2GU9hDsiv'
        'lGcyps+atjm9L7DVgXIkFOj1VtAScQqbfKo8Bmx+DQXXqgCtta63j5k4KTS8NrZ3YYJA9a4UrRQflsncTvdWDnak'
        '1V2rumusU/p3so6WIeiQIoumPlgERYLkliYmLBZZOez1u5pPh/Ksqlyphstj4WeaiC9edV/VF+8zF2772bE4txUY'
        'wJA40MhQKw1PcQk4gWEXiV+eN6M4oeuHEbDGaqmAK9NVnrqOjfhmjIn+VW5HeeKpIZAHvzuNkMgjw63QmsNuj5EY'
        'IZLmMWo5O6XlqcK96xSz4X1MVYBphTUiJ5hmjgKy9ZZSIi5yOhvWT2XCptYZ1zMKSmwXaDW25WToOQW4xqcJAeiH'
        '6YYOdxUWyV88rlSmOq4DcmzEelD82i0Rq00LNIS/Ei31aZGYK3u9jF2ZGwoMpk06jybB9ErzKnI9nULiFivQ5AA2'
        'Mw/Fr9sJE40r8tQwtm54dVlSykcFHMzompShW8Joas2SEddKrQTT6ToLphsffm44yH2Oh6RHIsBPsX63jIe7rHRA'
        'lDzFAgZnWmtSRqWFFgryKyOCp/DW43O0N0RKsGidt2DtjulQBDYLhUGbuzRTm+pjzWFA9Cuaw5tHTwJFsi5Md0g6'
        'AdXYB7V0sPbVVMbCSFuuM7xk8RQbDhSgiPgMGQYzjGgJdBVvbGLpXzH7DxLccepxnclRbU5wB0akBwzk1UztAGBt'
        '9SILwZpEbYt8wjYjgBwbHQCTlCNTrlky16vW2Jw7NV9GyoIk+0CfciRD0xb4Wk4ZnskCCQdqflUvZiJVEhdRrPFL'
        'bBkwgNEfsF8aihF7olNAfus0ndkj0bbOQChtYSACawwMZVu5Ccnx4Gt9PKbXtgK1cIOHHITgx5I3dd9OyoCEd5ru'
        'pmo4kiBjibN8F3zUq3GmEQKoKovH48bUIUxDLB+VhZocSN0ETaT+uQmWEV1joZVz4mKe8E2QVRkmOEfkYf8Fxllh'
        'OdKklTl95C6XdfCG5qeN4YBuWYBxn3pWMBnoh3nOM1zSKofmHrp+GMBqJ7KqJY+dDu8VKQ8uE72LohoxUHsyvD+B'
        'H2gSP1R55DD01Me7E3ZpbAd3eNvQR9GBDT+5tavgJVThFQtbHNPdLqLpYmjRtQ8LzDi8bFJslnxI2e+svI8yBDty'
        '3xUZBUPru6Mf8B8ACJbpIoDKN/uPYccrIib2QQ35Sw358RH+oyH/vkKep4hjZBVJao1pjDdRHk2W3FbZnhpQhptS'
        'K1gd3ZLPFCBRYZfXUkyMk6QowHtvh1RI8X6Ln9JdKhFlwGYLq2SHfCsvKg3feODCKk4gJ8uJ9cVZEk1kNbt3+uGJ'
        'cU+OIrbdri4Dd8sIXCYQ1a4rzSApyk4HMzfKIgpivujJ8WJDHEdu35H1YORvAOWYfjIaDPpV0CyYjDq9Ma6YO3wY'
        'SCiMyPho8rqifGzQJrqA2lq3gjf2CFfOYC4v0d1QXC2YlFl1eJdVGqig1lerINsAsdIj2RAOvERc4OVh0AyUG+Ky'
        'lGdoAMn0YMHRWTRHRYs7wLIA22eCqHMbimmC7NfeDy57BUsAXZznsZ8oDigWWF0c1GKQtDMRzSP1awF8FMQgiajd'
        '4GVMBxX+Dfr7uW1JYaC7cVpiFKb65MgqMfQHERAae+hu4E1i9CJ5HSbnBoS+e0kGmfofO4Gh7W4ELnzkWrAFVgKU'
        'imtWI6T8gEU52qPo4CIbMPBj3u8pW8PEz6Ll0p/w4paD3gOJ23RgyjYvuFIK8qavUgi9lxSykx1QGnzX674xVIrL'
        'vlFIUc/dhp5wYqmHGupKN1WXkmggeEaCd++G8pqVUVje5xuWV/1ERXlFUOGvigV9Pa+nk9sH2WqMqQearxr0D2/K'
        'ITVBX0p9MsQ9TvowDe1RrTGCta23IlGABIqd7H1m9u8cqwITS8gWP1UxrShbrqs0CIffU/xG34XUIhe392x5e28o'
        'L/WB17Ic9l22COJwyaVj0ld7wHoV52kwxQw5unMpoAp+V2BfdKNzQmzAzkRRX24QeCm1QKXvL4MNrHhbVCN9WCXv'
        'rdqVLphMkjsfvKgFBy1NLa0tsB6mBPr5ejaL7mzLS8OZ5WxrjzpkusR0RsBQanNa5JgkGM6k1slxJckTyLtBqYrz'
        'IrTDMJmB5GImhXRa0SK4c8Q9EPEM1mWvFH55gEuKBM2yGezDyTqdbOxKrbimBgEpRvUiFaEXzOd2uVJOhrZ1Ag0o'
        'zocjlV4FFMunWiUsHlEHD2aVvKZOtfJZA6jWpnIYUXsBsHoHWNJnaNoqkJwbADk3KovQbA4Fekfov4L3W3CwZmVf'
        'WpHWnQ5IPRpg1KkBQv2aqGpdG06z6two1Lo3gYmAGiiRUAMjIuooa2RUx32YnqjoqJXmfKoR06jc0kDZVg7mpoKk'
        'RXHI7+T2ArZ/JlIWNemshHO7MP6JovHnmPk/55T+9nnaynqpGHDDn+Y3tvxew54ly/3JRu4VHlRbeOoGbYearSzn'
        'roFAlvsB/s75tuYq8q6MNNlMqsZVcMX9jGPoEZSj2wAzMlOl7qT9GuMpJbes70RGbgdsk478OgYzru9G4Hx9Ex8W'
        'yHlMFzyjYjNgBVi2bL+LF8d4UOT/oC7LC6zfseM7MImA13HBQp5H89iE6LCjoAiA5YOKAIw6QIlIJO91mVR8dFFk'
        'H6opLTxHU0tUuVhaFtN1Pc/oZQbdvBNIBuz+L132l95lEsXm1QCvTthXMaIBjo5GKYYI7jMvx5R77DiYLmQVHhoD'
        'mRTjkJm1EYB/39172RU5uh1U8SIzGJNjZlGBGfKYdLl3+u4Qi2RqQM4Mg8+lwWu1mFez2TpDUi80hiRtFvQyB2Sz'
        'DFn/d2Ccw1/P85CPv2u0eS8/BZNGKUfBGbCvP7+lDC6R5s5eUMo8jYL98uVnBnzovb7rvcZkLLBIkCAXb73gFaCQ'
        'TTaSE8SEWuYp4AI2yGRYTAx7UQNoUPdJhNgHrNf5/Jkynkt15qqQfKcKybO/q5QRC6N8ChMXxAiL/P3rM1bpE499'
        'xDtLCAWNJzQO6SDBHANDwOQuWih6hzFbmNVFcitOzf/zP9jZMVCVJbCmigWvyVNew/APMayaL1LdS9VgLiulT0TO'
        '0SsHFQuayGFyi3mtlfowsBLeQ8kbnAaZ6Qpmq4FdeYNeGt2AuS/wmcZPaWoOtR1IOFpDy9hwLOnVOZLYvklsC4W1'
        'CYLZn/KKWoPSP5HQ+o5mOe0cbSHSEJQ/D3XNvfEJ+kRwwMavC+1Z5X0x41NGmuKWW4a3AuPGu81gvfnoJdgwQkso'
        'R9okYN+F/pMQHQprXcw6b8rQWbaO/UoRqgtmeHstovzLcpsDVeGj5srC3JfVtNXJzY0qquA+XZtC6spMSvSqZbPm'
        'ZTa8Z4SRh9AT9+Rgf5XATj2ojKAtUeWKAgRAzoagKmxLltcuUenk4Wk8thkBLEbUZC7bC5mLWL/7MsJvQa34mIGr'
        'EoIqojil7MV5YFlymx8I/Go7uS97e9F72HbjpbxEQjdR3NoFu2ffHhSzF3q4Hb9Hl9O+txCjNVCILXJl4Z36e3Aa'
        '9oxAuAriaAYKd5s5I+grT6rvNXpH0XigchzwDE3e0gFOOyLBOaolN2r7t7zbtxK3vO+tUJgV+JWchmXT2T/BT4rE'
        '4rZibolOtds1VKn8rQGr9wXV2isAaG9lnRhhWSle8QNQyTqDFY0fKqMLrNWlyge5ghU/5RAwShd4+OE+c6liiReu'
        'V2luIwSu1ByvjAX5NIpUuADZHxfDfutC1qXzoxBKcS9LGGADJuPrgisPrrLERKlhPVU31sJoBfrtdnhP0lmeOrbJ'
        'L04pLqVyUkt5d1n5iTNHzz2kg6bqc2cvsLl2SyP1n7y2eSHOTcXdE3Wj6I+8qjVs3NeqjrdkWt2G7scIdU/ajZKF'
        'XfVpkSplONuNMUVY8AI5YQRFq9MezAaSnxSRWKtOKZtHOxwXo8s0AO3ITIfjNTgfr1QmuK5VboF4rfIktma8YJ8a'
        'b4zElx1dXeOX7ejObnUZ9zvT2H3+Hdz2RFtS5W2J3waNL/C6IqaAPy+de1clczvaYJIUeMDbiWQdoN+ErU6rXVHA'
        'l0Gay0xf4/YxSsMJSoOca3PLKnpP8KU6Bv8NacrmRzNKCtVo91C8lQjCCtwydHMPbDtRrIW0azdOTX6VCbrhHZ2G'
        'V7xTNfLVaYha82ODQ2Y/Oh0tpGmiWvpc0mbLzRvh+F0X43uFgxojqBGm1dS/d4j/fbDMie4+Y6Jzjqkn7anAKPFa'
        'MrAu/n0Sf1QrQ8o0NDtGJ1oTzyfl2wi5AD9lU38p8ghq4v20iFcSDMMrE8yfEinJ3WaqdrskNcB0qhWgyYpdGWk+'
        'wfByk2+l0ElKSNi28QU/mSlFQBcxefiFzd3HGoOCXSShgyxU3XpgHKwM+1iGAFEn/uasosa0kftCPCfeA+KSNvr8'
        'R5SaR3vqnmbuGjKCdJmDasmbUBlTdOS3PcMO6SZiHtu/mhMuzW811/dkyQ8YWSAWLhO00WhDe45BKWYEIMQDlCi/'
        'blDNqOkDDhQzrRN4PiHDUxwRDEhHykOBgTQqeJtpWuU6oNWBEGUMdyAmx4jBDiRPa+HTATCZBt2IkA7UHGnf3zA8'
        'BeXEKI9AemAtxr9ucTa4OGAzUFI5fh5Ai4+we+H/7N2XJuH2zwMIE3Dg9ds/ERDOhK/YJF3ki4M7KzPAqy/s7hKP'
        'G0fVdg/2PqypNL15fyacUQqMwvlkmsux6vxeNRE+ocvmSSGcRUD5UH1UT/9oUupFOTKvwBOy0UiP8ptR+sbpRnO6'
        'x/rZtuM8Qbj1OYk7ousyOhbyggZgKa43XEUtJrDVT1xwyvi8r6IvKBW+kAoQSkqTqSqRV5onB7zSAzeKp74Gpsp0'
        'uFjiVzDATendx3KNGWhR3la0pkxgkSRSBwzumoAUM9AApXZIk0h4j03h04CVCQJwVZ6E2SuMxQ85TZD8nu1zA9b/'
        'D1FqdN+fiFHnnh78jJNYyJLS7OZHa+sesvgmt//ul+N3J497yCRNv9FFlgcweLrS/MoHnb8AzWAj4L5XfhsrCFfo'
        'GFhW624xMBnHbp512oLRx/NFlLM0mF7hp3cwu5LuxwfqHKYjJ7aKzHnsQyyPC0SdS0Fo8V2e33ZiYcbk/xwygSMt'
        'FWJNWZBWHGhnTuy//oX191U8W6wcLHtd4oOJ+H6/28VIu2XpcVL8Hwh8Ovb/95/++b//7Z/3u//z7//qFXeFKUVi'
        'LrcJh/puzh//uXm9U/HZeqnYFcoq3FkVreNlFF/Zj3yjS4Hirm+5VfWHU//o+P3Hw/Pjo8bnu77NxIBtGoLvYwvf'
        'p4TYKQr70MJo79NBZbEevXRjmTc7KcSDHPey+TKZ2Nau5TTufeMeRj3bLWZgSWFaEYXBs9TLwDjBr836RUKT6olP'
        '59eSWxVXYNVCVz4tYd+nzCrfx88Q+L68TVc84nLVVADGV2tRcP3E2NVO7Y0MFhF4a0vA0zZuZp3rp1mUXuSyZ2oT'
        'r1pi6zikU6zZjFO2u7nk0EqS/yuCbacGigL5RWaVAlhZVe0jMewOZrWd62wfT+spUH0kW7TIM4ak0VYblWaAtA+r'
        'ZkAxyzgL2j4i88jo/28ol/n2EfxxuQFCnaUUZDT3NCNQfJTE9MG7e7v1+1zO3mthf2OosxZjBh00YPffUjRi/w/e'
        'XBBT1GYAAA=='
    ),
    'cifar_gradmatch': (
        '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'
        'jG4AAA=='
    ),
    'cifar_spot': (
        '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'
    ),
    'digits_kcenter': (
        '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'
    ),
    'digits_gradmatch': (
        'H4sIAL6IJ2oC/708a2/kyHHf9Ss6WBxIakfUjHa1D8pjWCvN7m20D2ElO3caTwjOsGdELYekSI6k2YUMw0kcOz4/'
        'YMdJEB8cOAkCJ3D8+BDgYDswkPyU3O6dP+UvpKq6m2xyONrzK+fzidNdVV1dVV1dVd3kOI2nzHXHs3yWctdlwTSJ'
        '05x5URTnXh7EUbYim+KsxaZeftxieTDlLfYiSMZBCA/Z8SwPwhY7yeKoxc69NAqiSbYyRsoJIITBUJHdh58rcWbz'
        '6CxI48jOeO7zsTcLc9N4ut97cu/R9oH75LOP3cO3n/W2dw+MltExrKUYj/c/OfDjvUcNwJKvaDZN5szLWJSopsSL'
        'fGiAfxNftcH0kzDOYUIr5aM9y7hpbE8mJTmtM5njE5EJcyGT7HnIQUi27+UecJgp4YSx57t+MAnyrAo4jX0euhkP'
        '+Qg1ouDz1AsiN+dZ7mZJGNSoJylP0njEswy0oVAOcpxV6h+MvJCnNXb4KAaoLNDH2N/ZrkKFQQR/XWJJAT2KJ0GW'
        'B6NnfJLieHGkcIKpN+Eav1E2jtNppdcecw9tT0EdxxPRnx+n3POTOA5HeTFU2eiGwRQltdBiij/djrWibNEGQ815'
        'qn6aRjCJ4pSDwlaeuM96+73twwPWZZvtlXvbBz33oNfbhZ8b7Y1bK/c+u/ugd+g+2z58+BRhQpipGSV2Gs8iHx88'
        'mNWEm227vdFibXuzg/9tb1gttmFZKw+ebe+6j5/u9h4hct94wdPYDUFeORggKZD79Bvl5+4A5o0lHZ3G9iWt7cbm'
        'G21jsPK4d/j201330fa93iNg6aXOkmM8SD3/sZePjhk2r72BUx2eQNlO98opNCMsm1oT9NIJNwIvlUMTNDRfruw/'
        'enroPn2223tG+tpPY1gS3F87ePb00Biw66yvC7B/MWBg0eyCBRHTdD1Yefb06SFQQIdnGuvTKF/H5b4ulrebpXHu'
        'nmXuBJiYIhNiQbnnXpa5JxnY5PajzTbY5/2HD9zdh8gLElw3xsEE1kpmbLFnvYNKD7SCn8uMlWvsGTxO+VrmjbnD'
        '/JiBK2dAneecYFkwBu+W5oEXMonF+AXIxl7BuSQ4lz4CtpgcvqVGGzgssafP/SA1gQKPYJkdpjPYAwjfjZ/TT1hW'
        '4OzsdLQPi2Oa2bME5s7NlysM/pFTsP0kMBzWudNuMSPzzjg0y7abm9g2jqPcHnvTIJxDG4BE2VrG02BstCQdBNCa'
        'HVDXdgqTAp2/zcMzDrr24PlRMOQpbWXsAKChZZefeJ+biV+DCrXgBQc6d+1NYMC74JkdekMeyuZOu2zPgzzkqr1j'
        'I8Mhn/DIt5GQ7LgD7YK6oAVmeB74+THiEAo1c38CnjeMU5zmtQ36x1CdY29Udp4fBznHLinCpk4xHg5VHXDDbiPz'
        'omPqpc95KrncJFYuQFrPbVCs2GOQYhDhWPPmDjGOwJp6J3GqhHfDvlNgNXRIjCAqOzbsmyVGvUMfp0lK86U9JL9J'
        'GvjQft8LM67pCMyS01xEhxgk8cekvXye4Og3wZ8bSVZrWrm0VnaePnr67ICcZ9U/OMa1W9s3du9ug5E1elIAaLfv'
        '9m7fqABUXCeB3N64t7EEBN3itc1b92727jZCUP/u5mav3W7uR+d3rXfr7v0lAOQdr92if8AhPt5+ttdbMtv95fPc'
        'vWqGf3rF3M6WzipbPp94+UzegTk8evikd3D47qNe8zTWlk/DuWoaa2tXzGPNXjqRJXg0kyVoNBUH1fHwycN724c7'
        'b7t7j3vbTw5caHBFw8HDox7Mr7NxZxFK/Li/vXP4FLeMG4sQ8PuwBNhYBDg47O0fuPu9Z+7DQ9odOysrEF2z7NTH'
        'mGi7xe5ZDq2kbejEyCjz0tSbY4+PC6gLbWMIcvNbN60tdq8KdK8BiIjtCjjY0cztVSDlwU4DsV3fabEnccRxS5bd'
        '91bvld3Y12LOgK2h62OrzNxmn2H37ENBNeUQc0aIOQWMKWDvUtxmiTmNPdjNnk857BDmEaQ9kORw7nfBUUYuuNkU'
        'dCXnmkJwTQxCGOjHU1vmGy60m4gDMz2qzvSoURyQOfkAd2Rnx17Ct9gUfoAzNIMoN6cQTEaCb9i6p+xTXdZ2GIQ1'
        'GWef88IZ76VpnJrGlE1nWc6GnIkw/gwj3ALr08CFo01cBq6Rxg4MBsy0gItRnMxNgTyk+ARdsmQpwjzwwnyDJbbY'
        '1Ua4yqaW5C4K8gb65bgNpHRrLSlFbpbzJEM6QAB2FZSe0JgFYG8waY2bwL8AIqBDe3QcByOSEnLSLXhtgSSTEDbg'
        'Lm0hqGbsQxX2FYkBERxBmASbrTACPk3yuWmCQflWox1I6H57gKSQUh8ngazAX9i+0swsmLAsMQRIzfU3yoViCl7W'
        'NGow/1W2UawPQsNo7wSjPWEJkLlMpVXjP3mcQ4DYZcSbKYYg6lLYJCyS01UMou0JSmC1Hb7W2WAQUPECR8pXk2vS'
        'ldNZF4jacGo6J5pwQMxbFQHAMy1o0dZaKpOTukzEOCNI7HKpLdwKMlJWo2tC+bml/KT1aSIcLrUj3bjrhvTOkKxo'
        'WFgQmTZk3ZDpS1eSTnChSM/7zrClJmVVFayYHCGTgDeLgtMZMCFoaYySEfHpUNjpO8N+MVyXjUC+oygvLCHkkSlh'
        'LTRYkld/BI64i3AVmjz3GLWCMnFJFsAtjIA1cx+hRs0OqAdQcK1qHdeJzKpi0Ea3bNI02+UWIWVxVIpii4ngvSKy'
        '3UJAalLQNwwiYswUCC00J5jlJD/uTi3w26h8kyYv3Qw/1+zDDYPn3FSjFobhB9PSNHzLQSzctKB9sGTYcx5MjiGl'
        'OlJwVU7E2JDGCIl+mrW3iCg0Ddg6NeKz3BeV48fkD+cfhiZ0ajoP4xGt792+thmAAQluQOljLy1kl8Vpbq4RCkhu'
        'BsETypznZrOhnR/zlJtfwAHB91TtDOFwXQAkjFDtkzyjcwAQi5gHOBxvEVCqQtjOUV8hDQR/tuf7pmoDnocp957X'
        'HTI/b7SSRWOqriu5iQpMZSDa/imxRCwhU/7UO3eP44kry13gJLH6RS6g27kl9eIjH1od0ATGCQ7b/cwWz9qIZJU3'
        'NixYX51bNhjEXADmMBOeL7AmrBM4QHr9QWGspA1BvJTzRecWQBVVOzulhNe8aDGN9xYrn1FKUR64Xhh4WHWUtYEE'
        'EHl6xl2yMVkhUIOAWGAQGArJY+hjrmnO6xj6QGgm9LdYnAYgV1GU7kIumQQXIH434ak74mHYhQ1fCD+UrUOw1+dd'
        '2Nc60Ew/XMgup13j0cba23NIJ5hUhnsGyW2c1lgjOdleknBR6RvF0cjLeYS1jD7w3WLHA6seTp5luTd6bhKuVVcU'
        'Gcdchs1YsnUhaHT5Bch4lCuV3wf5hQAmYk6Y5siDrmn3pgo5xaZSLlp0yHPJiKgJA0SLUWFYwNYrxWYJQPbXtm9A'
        'aCsiWNjEYIJdMXaWY+VkPO/OrU8W7VaZKIaG5z7ugqCV6UyokNguelUkozMtH5sRZafCu5+jryLR9Qui4Anu57xs'
        'lzgCY04Y8wr0nKDnNchsBI3Vyrlp2WMQI4xqbcmxs5FdLJWigy92SAMDrWJ5cGcbwgaqu0dUTUPvIxXeQsIiHQAf'
        'utbRf3cGYEhU8YnkIltQnhjmiHgDisivW2fwiBjE3gYOs9HGsnkfSWwx7w0N+0ijW+/hlbUCkM3LI09bSGBJJxer'
        'Z5LCFOfu2BsFYM9zMONpEHopPJoHsNZnPni/ckPFDVEungj22YgUqjItAaylW6IBc65RkT5JIJl5aTkUZnT9wUIO'
        'pfAkByzIaFd2igZaRdCSmcjRQnwJ/jzKiAhGy6VPrqCrjFI2NuYTJYJ6WmcKQUTzEF9RXC61TqdMtL0X6YoSZ22O'
        'EEHNUj1QjppZ8M68INQnPFJgwzgOy4gpL+MlqYJy3hNYpHLWKmE/wFh+lhaFAJG/24eQ5sv5bQm0/heIA4wS1oj5'
        'cUH2RGYvcuuHCJUwMF5QgujniHgipyFSD4rUtekrlqChxQ6QI0gudGNXxITx4nqf5dw9nUF+k5l7sGeFkLhjnCFm'
        'jMk/ermiGcbykiJGIZujfquMuhRsYYswr8IM99in2LSU5mljeoMIW+y0r0d8OKzVd/YGOO3OlprP6YpGhpRao5Ly'
        'KQgyoO1iDxQ1XSk3+7JvVUxrnf4Ia9zCLCNW6TI9m4BWbKWSSUw2BKAUcxb4MwpncTRU6anMVbcgbvB5PZJFmgAo'
        'SFilDcrAlFD6jiI7cEAsuB/gsB2xTmZpAhIHsu2mXFpPpEH+p2g3n6ZpwpNTYAM56lpTXVvid1f9JiLg6ENe4EDc'
        'X9IegXfGUO5kkYFi3E8pYgOdJQytEdvRImNF0SYhPefzbuhNh77HThxmSiItRW0NiVuYvp6B2dXjuoIdGmOlFuCr'
        'uQh/WmWALAv5vk4Wp0DXlOQLE6S1lMharjhSI71RyU7V66YV5y/X1ZFWFis8s6q43bwj6k8YclKVgk568eaDnZ2C'
        'WCJMTzv2jU3Lkoaj8rcysVhSPxRBVVlCFOmIXLioxiJ1UkkJeB7MoOrapcUvUxMZAelkRotuAScrXQYYkEChdE7K'
        'rJoPkFlp2KUC9Vy7j2BgEPIBltpBUSXYRacIyw9Me7dABk0BwNV7NzEJ5Kxy+1Y+pujC7bH8Uc1o5XxU3I4gfRi3'
        '6o/VvFoVnLpFqQsXLojE3dP9dBVPWhe6wytc+5a+t2qyPmmx56RaHs2meFLJzdOq94BuXPQFvg0ZAyebTNAfVpmB'
        'ldN3np+ozESqXaGi9vccnZNinwMn34zxKcAo2NFS/wKisuiFt4Xp6EwtrQBQ9+9YAyjEIlWuJf2LhYBGCgtThnW3'
        '1+SX3ghYDw116S4GiTKU9XxMbF1wMWlwgf4Ccj+6D+BgAiZ8h0OKxvQVsusXlAhRb9dAp0R5raG8XO0oYRR6Wcbl'
        'jipLgHMQzs6oCDMIQMhm7hb5Zca9dHSMewH3FRAwJ+De1WMIrGvujJor2u/W6ks0AIUTdrtww3SdCIRZuY1SiH5f'
        'RgOzMCyHAvT1ndGy2mw1bN4500vZOJgtEmFjx7D6a52BVmMOxwC5eKXJ3OkiFdobhAO/0Yb9JYtD2P66RjgcT7Ci'
        'sCQPk5RF9lTIUM0Ne5KU+8Eod8H9DD3YoRryn/r2razM5acQpJhIRWrJbSmt14q8+1icmUQqvn+T9haW7SgOkXTV'
        'XekDW045BgbCxY5WWCFOF/c0+BetYB+hgOygyif2KDoik3W9mSw+HIsaS/+oVYSfOIVO0wykZh9gednch93oXTo6'
        'LE4PVwVhlUXgkaFVlKKAqKpG4QKTSygMIi+c2NhiPigKgyqNKuuu2lIl0y4XqkP8PMAkL5FBBZIvUhllwo1kwg0d'
        'v4J3FVoEAICYgBbKBbJ4iFhBrOyZD5ossvBg4hZTBpbIoxGXTkykcZn0ZsKRLYRiW+w5LCl5ZicxZB2pjWWgYMqx'
        '/DN2qViONYhCo1e4zmrEpU+q4jHFOFQsRYkuni2QCZKwPyPhqBnvD04/sTlkozglByyolWonMmIgLUsJ+QUF4GbV'
        'ca4JMhYYMEpMaRuicVrODYICc8/bjfsSjdZw8ivJCb1y8HduJ4qwqlMWZqgK08INXKpSDwmxB6ApWQLNolPDipps'
        '6TcdxAx0BoWHxpQaFSHQEZ9b0tZOMtcPwONOyMySFjsFphOgQyUMwVBSLY8kTfWV64i1pVJYBXq6FLQgnID6EpWr'
        'IvopNJyqBrSJtr2J9w0SQDy1FicH3avyTDLBB9jszGR9auFAWt+p6jvFPjn74SwIfXfKwdpHuFPkMXJbJDxg+7Kz'
        'e6et1RJQ4KqnJZddcT72CfOW6zfbN4vc5c7iWelvcIi2pU6xBDodZJU8qJM/PNYSKwOmnMducepGYcAUXXNDSaoE'
        'LpMpeQw1qFTNI7wfXp5mF6dAGoHiJKxSdbtioiXN+oSbynCVKtziuVKhlyIRkgcHswSvQrt4Pgv6hNhMapZURpX9'
        'OqI0haOMDso0eyfomgwH9UVdUDvCJKbpIFo/VcVj91FMAPrJYiVjbSkqsr4Be1IRnRU7olgGSlyrBWGVdUNoDX4/'
        'q2tFUi6zx6I6WuqJVkBmNZ3t7g4htNAcWqYfKEvaYuVVJYBo1UPC02bGBHLJnppGnb9CWHVTUj5HVbiq9VtpQudE'
        'seosC0GcSm8C7j2gm7lkBpCrQrxz5qULx0xXGIueMVK29kdYgYNAERvKdINSr6IPfuEtDRNTy3bRgOUCcUlJOxzE'
        '6GQb4sYUd24RoYyNl8DkpQMxKE2AlS9GwKDdl4qVyxYTo4umGjPQu9d9uQd/ACW6NKRIIK+dJS5eHYUoQArCu8AX'
        'SVzULazL+NykStcWtuP1UlMcwJwfB6PjrkFXXiEPoIu2+TzkXbrux4q7uF3w8ZsU8caQNlzbvYv/AwAvTI496Lyz'
        'aalRG6jj9dgqdadG/IZGvLeL/9OI3y6JZwnS6Bt5nBgDmuBZkAXDkJvqHooGlKLlNILVyYV8rACJC7O4klulOIzz'
        'PJ4ugVRE8W6vm9AlchGUIdqxUYhD/iouaXfv2LehU0oCJamU6orAkpRYavZCr6hX3vSgGl+7rev/IgymJlopZH3C'
        'VUkrttYw8yiaqOx1vSOni4g4jQzPzUl9WqEUGMfD777jbJS1F2+ICSkulgt8cCQUHrq6uCW1RPugwpsYAnprwwrR'
        'mH1cNM7k0qAk7ozKM95wIOWDbyjJPcQElzL1UoyiZbQwJxL40leOL3vFM/Bg9BYYhJroo+S1JSHQcTBBJ4hhUpiD'
        'exoi6cyEZlKPecu+22I3YQFg+PHJpE8Me1RRKt/NKF0Eun/Jc1/9NQA+8CKwQwwe4ceACtruGaY5mWlIW6CXAowy'
        'x8WrExlKSkz90sU4CCwUowF8PwzjO16HyXgFQk/RpYAcIgzTWZ0LfHzkZVUAbR9axaXyPnK7xYIMtwkMN3HqmL5V'
        '7ywX2KDrcRCG7pDn5xy8HBjZfA1D9OtcuQH6b+kCOjeouiYHoLt4bbt9p+JEWuwFJQj6PTIYCZVJI9RIl96ovGhN'
        'E8FiCb5p0JWXyCuNxdsL3eLFBtFRvBCh6JfNgr+O3dHZ3QB7WphTB3xdOem7d4opLYLekB6ki3ufDC0WHEa5rAjW'
        'NO6JE2EyIra3/oSZb1lGCSaWjSn+lM20iky5lhLP796murC+6ah1Ld5VMOW7Cl35CgMEE2F3o8WOvcgPuYwXNpTX'
        'n02jLPFGePeG3jARUDm/yHEsen9lSGLAwUTThtwS8P2bHN28G3pzWOWm6Eb+sEu+omOW6384jC9cCG6OOfhlwjSW'
        'wNrnQX7sZrPxGHJ0w078sWEtw0e/MQrB0tFpKP9N6xocCF2XOU/BWEx/LF1OhktKHlNdOIUbznLf9P14jKl4mW9j'
        'IACbv6VClgush3eKVSBP9siL4D2qMWzB8SwZzs3Sp7Sq7mOAJcBUeUHbm0zMYsnsdU1jDxDGQZrlOOXIpdsm0Cyf'
        'ap2wikQfPFS7ZPWMeuWzBlAu0tFolnqjObkuAFa/AZacGd7QUiAZrwBkvNKZ+1V0aNAHwvgSotOcQ1Aux9KatOF0'
        'QBqxAkaDVkBo3Cqp2tCVoFYNXmnUhq8CEwM1UGKhBkZM1EnW2ChPh7DwoviotWZ8pDGz0LkEQYVVVIgESwsin1/I'
        'vSUGjrwwrFpnaZzLjfF3NI0/hOb/kCr97fW0VPQgcNjoR9mZKd+GXDfwUmcKm1XqZzZ0GHhiDAjFVW7pSxbwZLs7'
        'nMt9phFb6noBW7a7Hv6d8AZcUa1qCus0tTLjEBN51UAbGDTu0jVEh1UiDnaW2ax8LWkWYbHSD8Zjjm+BFoVYUXvF'
        'cxD55ui6Ia+/1t90VaPaSTQxymCyjI2bua/YEDP+pPzJMF/3SP/lHJr6F5gvp/WbT0Pjh2ZyFe81K2XGHx+waoNi'
        'u9Lxe+X3JNPYpLgND6OLxWJcY4KHhkHWxCFdxiN6tyjI5w7LQV1ss42vMOCF18+rF0MFrWusdwGKhQUW4QvIWTCJ'
        'qhBrbFd8A8FR3xeQw0Og7MHQgiye4mbMo2v38kJqwH12u71+oy2u0dp1ss8EPw7yRjwKSoDFC47AoHvaICmELDDv'
        'jO45ru/vbOPNFnmcn1UXA15z9vVedMhLBSZ9c7bAZIlBKS6EaalTUylmPS/LFxJbTL7MztTL7PJlRLz7BjEx/gtR'
        '34325cJgMlzFkoJD4WqXbbwFuRj837btFgjqrQWcx6Ks6LDO2pMndB+3cBYtVYZcK8uQ9dU2AsnCgpsLcVVs2maP'
        '4nMyakQezqh6I6uYzEs5g3wGxLHA0c4sPQN5Z8fxOUNvzf7rZ+ygB1ylcZax/JjXFF4X+ucjMMqncguV7pOstvC1'
        '4rbPTQudLuYdfnyOt0JL11qhRvS2lQMdztU9UXClRFWtfjsJziB3EnSqAWQRt3e1XVxkql2jsmkb1iqkjpZkcaPK'
        'YgNfjc6v4JH4+x3Zq8cChtUsvwbmqi7u98rVYjTxBr5EKcUU31FY4jg1twc7Plq+PYV40KZsxMUMy4SpGfZJHEQm'
        'OVY8TYNszcdkzJjl47U7RvVOqNoYpOmJXCedRe4CC2jRm21TZjzqBQFXxoGf6P0SOTKFKuX9I/UZiC5rCmlc2U0B'
        'hsiNco9OHpLi3mP5LQmbvgGRmRYtd9WIFyFMy5ZvGlQvLyYpP8NSkW/jJ1woxpFolToKJmoIalWxqxNCANQt3qEw'
        'DdluVC8w6NzjhSDE6QMsFkDlbbXr8pJhySICjo1+St/UGNDLOeDqqKQsR7EuWRqfw55F9PFeKw7xshjteucSXzUJ'
        'Z9mxFt9gXQvhijt9BXyLFR+l0W5v4Bkc8F1+oOY6opelcp64V56Xy7cBaue4WM5604so4o92nK69kVK+nyjP8+f0'
        'AoA8xkdDo6uULfVib3mhMl2N8AKlkANKoVIBLG+DuPgCUYwhsDqGEj/LE7Wlh6Lq7lR38Wy0IH+turWXb78ukeXW'
        'kjuAZIJNV1ErTFwHB06XUj/ZRdNVdc3U0tZMnMAk+ZsO/CupT9b02QPwAZdb6sAffGKyDKhyvW8PVSXVW12LyyUm'
        'TpJ+i3uV1XdJi5sOSgbraHDqxgishzcKBP9pOvSqzbrmNQopLnwOqLhx6GOBUBPlUkgJU70yOZnWPh/k1DimHQ4m'
        'Xvn00GQ6qPF5WtzfcNXL201XcwonUFzPmVQO+jfARpeKQDIzoD2kOudK18LNseWG03jYCOw235msYy45xEROFoEb'
        'JrJch3WIqmxXZalxz5LvNitDrOpWKW/J8YV48To/jtGua0aDe1+h+8VeKlqWnwoxFiROoRKJhESj1g9w8iJIqgcu'
        'mi00iKjVKBZrUYsQrsI03nhtaKs4nqbDm+XXGRCVeL9qB1jUsqrISL29pC3ewZ3VQAM3HNoPDPn9PfxYDW14VAFC'
        'VRiO+NsqgktHPtSKv44QsbFnOHutoqbrRC1Vw3XElghZg1a3darH0bhjQn9Rl3HgqVUpdDgkrlr9wDnJWk1lLEfq'
        '6lLbM3wb8+z7mGQWQYuqKsmIq6F+pMc/9aqAw8ZBFGTHYE9ausVeinBn/WURwVxSCtvgmddExOLYG+PLrCE+8sci'
        'NFzkXNx7gzxe3mAvP+G3SiHIwlGi2YHNAXs0F1u9Hu+P6XaCIuosu3nQU6O+VKAi9gMXGuciKARSl+VnV+TlaX+s'
        '4kypdBQ/fUpQXIugRxEGlrGftYQNg8JGXyTKeA+4bds375Zj6m/8g9vMUFN0YtLv61XfatV2odq9aFkD/aBzOXdP'
        '4mhNDFlk9D7PSViSx3rmUxzhLhzxiFtmx5yu5Lwsk0i0OVfYHMQo4oqEgYowHAqlpCIgBTOUolytW7VhfyTpqL5S'
        'VZHSlZ6+otVOcU3WgMVNgJYE8C4WASjBAADpQWghANSixQKMdDcFUJMBV5mCGbiQP6Lc5Vfaem+soJG7WqO6DiUB'
        'rf+HIphejUEYV5YuHT3+0WBkWE/4ZI7DuVt4ZfbyzHlp6IdWLpoUJixS3WKPnmiGP9CPBq3KMriSgL5c6jQWjo+W'
        'UqkusyqdSxEwtCboMptOdywRil/KcoXK2MXHIt2dt3s7ezZ+hLdalsAW24dVm5m0krAskeFHD7xsFATqXBm9f5R3'
        'N5ZWLabiWy4vm7ZNbQ0V7qtVXaS0Riu7JwD35WmtVX6EopoFtmpGotsIM/TqoQoagOiVng1Jqq8+JEHCsVQDU8HX'
        'QcX3JGClvf30ARvB6oElEgPOFp5OqJVUfR8egGGRyDfxMfOH5ZMTmlhc2ERLy6grTUoR2fauVhpC/JY6kyUt/ODx'
        '457767/8+qt//vpm+/WPfmDnF3mtePX6/R+9+sbXPvrlv7567y8+/ODn6hzgww/+qnYS9OEH39DW+YcffP3Vt957'
        '/Y8/efXzf3n9wx+8+v7XPvr7P//o/X94/e//hLVYGEsM++rH3//1v7330S++8/r77//PF7/0+ej1T74JnUqsAPT6'
        'b3768a++9/EP3nv1lS+/+tmXZH3/4x//+0e/+LPX//G1j3/y3Vdf+Tb0/e8v3xPQr97/oej99fe+jGyC7GvqQQ9W'
        'gH/8q28B8dpkXn37vddf/eLr97+qTenVN7/0+rs/ffXNv/vwP79DrNZO+RiQQYCf/zXKCSb533/LNjbxv7dhgdy5'
        'vdkmLOMN9T9/LCp+U+85dzEilwE1PNKtieVfj8VV8Bi/Y6ZrFNDKEEDRKGpyTtk0i8Dmn8uqEN7MUF/yto+C5D78'
        'NRUonrcZrbL74b6727v/aPuwt2vhF61fjMsU4MVYWJNJPLsuYrguHiqnowiit67RWNlcLK4Kb2Ync6OaSFGlDO3Z'
        'TidhPDSNVcNaeK8Ywx0ammasWEpKLsBhmImdQoyMn8eDVI6WiC0+ZmtVv+aixLCyApRdF/FdlzIx18U32F1XvqKW'
        'Lyt8UAx7ZUV3i71I6NUXZQMrZdRt7MYRfiiNvQUxMhbQDPYWM80lVQ5r/VbbolcuCBeUhapLrJX/AzOTo6PiXQAA'
    ),
    'digits_spot': (
        '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'
    ),
    'self_m_all': (
        'H4sIAL6IJ2oC/+V9/XMcx5XY7/grJqVyZoacHewuCIpcaFUBAdBiQIkoAHYc7G2mZnd7F0PMzgxmZgEsWXApKefk'
        '5GTfVWyXq06OL85H3cV3p1wqriuVLPuqkj/lRFL6Sf9C3nvdPdM9OwuQ+nB+CG1xd7tfv379+vX76g+O03hqeN54'
        'ls9S5nlGME3iNDf8KIpzPw/iKFsRRXHmGFM/P3aMPJgyx3gSJOMghC/Z8SwPQsd4nMWRY5z7aRREk2xljJgTaBAG'
        'A4l2j9rvQVd7cRZc4E8OF8QS5N48Z9mDR7x4GIchGxIZsn4rnkU5S1fizGXRWZDGkZuxfMTG/izMLfPR3s479x5u'
        'HnjvfOdt7/Ct/Z3N7QPTMVumvbTF23svD/z27sMaYEFaNJsmc8PPjCiRRYkfjaAA/p+MZBkwMQnjHNiyUn51Zxmz'
        'zM3JpESnVCZz/EZowpyzZu/BQ8mSB1N/AvNAH48SwfnsJGQwFe7Iz30YQcG/MPZH3iiYBHkFcBqPWOhlTDBcwuep'
        'H0QezEnuZUkY5HqjJGVJGg9ZlsGcyyYHOY46HR0M/RBmSieHDWOAygK1j72tTQkV4CCUvqNsHKdTByUhTjUgd8x8'
        'FFoJfBxPeH1+nDJ/lMRxOMzDAlVR6IXBFAe/UGLxj27LXpFC7IKEg7DJn5YZTKI4ZTBHK/c2D3aMLkm0Za5Oo3wV'
        'OQ0173j7O3s7m4cHULveJDjvYGdnG362m+3bK/e+s/3tnUNvf/PwwSOECYMst6LETUGwR/jFh1FPmNV0m23HaLrr'
        'Lfy72bYdo23bKw8fbW0+9A4efWd/a8d7e/N7gKF1u7myAoLhpsM9aDzN3FkCxDDr6YoBf8xxMAE+uaMkMDtG607T'
        'MczMP2NQLMpurWPZOI5yd+xPg3AOZQASZY2MpcHYdAQeBFCKO0bP3EwDP4Rl8BYLz1geDH34/jAYsJSUh3EA0FCy'
        'zR77353xX30NW/CEAZ677joQ4F+wzA39AQtFcatZludBHjJZ3nKR4JBNWDRyEZGouAPlHDvHFUTsPBjlx9iGmlAx'
        'G00YyRMO87U2/TFl5dgflpXnx0HOsEqwsK6S94dd6R223SYSzyumfnrCUkHlOpFyAdw6cUdBytcbYgwi7GteX8H7'
        '4a2m/uM4lcxbc+8UrWoqRIsgKiva7q2yRbVC7aeOS/OlNcS/SRqMoPy+H2ZMmSMQS0Zj4RW8k2Q0ptnL5wn2fgvk'
        '3UyyStHKJay2rUcPH+3jahEivZfGoETYqHGw/+jQmHa3Vm8RMXfpj2TWIlSboLbX13eazaVQhkVCaAzDWQarP7Op'
        '0e3Nte27m8satbcIqNm8u/P62jKgNQn0evsecGzlcuXtzf3dnZcYWHzNkLZfbTB7Vw/jX109gIxIf/jgnZ2Dw3/5'
        'cOd66jvXUN9wX438xtXkNxpX00/1lyt7Dx8deo/2t3f2pRbmUuaesHlmgapdefvBOw/ubR5uveXtvr2z+c6BBwUe'
        'Lzh4cITqv9W+swjFf9zf3Dp8hKjXFiHg92EJ0F4EODjc2Tvw9nb2vQeHRF9rZWUFvBAjOx0hpZuOcc/u0CA3oRat'
        'RuanqT/HmhEuni6UjcHY57dv2RvGPR3oXg0QIdvmcNlsam3eAFT+RYAGsddxjHfiiPWNm7L63o17ZTXWOUanbzRQ'
        '7Rk3DGvT+GfGPfeQY00ZGOoIW06hxRRab5NNs8Wgxj54FydTBubBOgIvE3xKxkZd0JKRBzo27baaYrApeBlEIdjI'
        'UTx1hWPmQbmFbWCoR/pQj2r5AY7qCOCO3OzYT9iGMYUfoAmtIMqtKVjaiBMejKHmja7R5L0TBX6QMeO7fjhjO2ka'
        'p5Y5NaYgm8aAGdyvOUP/oGj9JlCjtC44Iax8pJAHnQNxDlA1jJO5xZEM/Hx47KF+FiRG6IZfWNfIpmNcLZY3jKkt'
        'qIyCvAZ/2W8NKlV+S0yRB0s0yRAPIAATg9zkM2gD2DVCrlATjC4ACcypOzyOgyFxCSnpFrQ6wMkkBGvcJXuC0451'
        'OKU9iaJPCIcMw4WMCwWbJvncskDARnatXAjoXrOPqBBTDweBpMAn2LI0swoibJt3AVzzRu1y5VicloaCDcZ/w2gX'
        'C4aagWNrPIYBG1wSwM2b2qWk5BB/hYCTaLN4F4RdMJuYRXy6ikCUQY4JpLjFGq22Ad4VK9oI/ip8TbpiOKu8odKd'
        'HM5jhTnA5g2NAfCdVjgvc5by5HGVJ7yfIYZ3YraesDTOaLJqdRXyzyv5J6RPYeFgqRypwl0VpO8NSIoGhQSRaEPw'
        'AiGQUC3pBBeKUMXfGzhyULY+wZLIIRIJ7WZRcDoDIjguhVASIjYdcDn93qBXdNc1hsDfYZQXkhCyyBKwNgos8as3'
        'BM3cRTgNJ8t9g0phMnFJFsAOusOKuA9xRq0WTA80wbWqVNwkNDckgS6qaYuG2SxthuDFUcmKDYN78hrLtgsGyUFB'
        '3SCIiDBu6THLEEQwykl+3J3aoMdx8i0avFAz7FyRDwjeTpgley0EYxRMS9EYKbyG1mjNoL6/pPtzFkyOIQ48knA6'
        'RZwGiG04Z980mhuEFIr6xioV4ndhMKUhiCB8Rz6EoQWVCj1hPKR1vt1TjAIIEqcGJn/spwUPMwhlrQY1AQ7OwLNB'
        '3rPcqhe482OWMuv72CHoIF3eEA7XB0BCD3qdoBmVBIDYRDzAYX+LgJKpJENHPdmoz+lz/dHIkmVA8wCC7pOqYmbn'
        'tdKyKFT6+hLGlLeUgqLYUdFKOBmTFPyDuQchXBAG+RxWP0S6fgpfrQPHGMwgKMzLycfJE7MUgUxEQ6DsQHoLHFhx'
        'GXgB+g3DwvQLoIr3oNh/9E56/QX7L9sLSowgI0kqccgKYhXUZBZSuKAjYc1HGTVCjb+kufSSRGGtTSwbyG+rhmzA'
        'LRLoCLItnHaeQiLRLEyuZG9lrKAFZqmq7KN6EvwzPwjVAQ8l2CCOw3LV5+WaF1NSjnviB5EYtfRCD9AezdLCu+VO'
        'qXsIvqsY3wZv1vs+UYAS3iDixwXax8ICC7EFLUstUNYlI3o5NnwshsHNJ1kbZfiSJChwjAOkCAykKuUSmZBmzKLN'
        'cuadzsBIZ9auU0ZIfMjo0aKtKIqhMz8pFhgJH9XbpcqQsIVQwsAKedw13jCmJTtPa200NtgwTnuqusJuIYLY7eO4'
        'WxtyQKcrChqa1QqWlE2BkwF5/LswU1PODh+1f1l3gw9rlT64OG6gqYylz0ffLWhW2BJBJFpMDij4nAWjGeli7A3n'
        '9FQ4XBtGnI5YVQ0jTgDkKOxSCIVWpSa9jkTb7wBb0KfAblt8oczSBDgOaJt1DqHqDQL/T1Fw3qRhwrdO0RrQUVVD'
        'Vm3w3135m5CcHwchK9q8qeqkIYRSAN17vEhA0e8bEllfJQntArbuKGpdYnSJSRBKd0N/Ohj5xuOOYQkkjsTWQOQ2'
        '+mBnIHase5jOmG7KiBzqY6VineRYSMFWCCDJQrpvksRJ0IbkfCGCYjEJufcwf+xxe2LNBfs1wyTU5Vxbmr3C1EpY'
        '4D1a3Drvj4NA5FDX9UIMrNHgKDZzSciMf6lRM7e0og+c5BpSHxekKnNPqkFYXRHmyNEOFxUGzodQJiBacmxJyqSa'
        'IpfFoyVB9MqGFGt60jooIz6FchRhXc2pwIVie7nEwErhbAkykBv9ct05fB3AoFg0m2L+mlkVlaosFkUWsKTeiEv9'
        'RkkF8BRwTCT9xCpoZquxHKw0kmVQHCUAzk9TF311DK6fJEzuGizxJyhIiPIgmjG1s7KLN43qnoLeYZYO9SiKj5gC'
        'qWrLajhVINJ9kBIt4iqK1UiiB/UwmKMeQvRLRAdFJLSNRhO0M2i+bRzRNmWxeXy7XcAnMc701c7faen3SVuE7AEK'
        'bHSjiu8rV80B0tmD3voLnqjurSpNi3XCs5Tl5psHIuntqlbd0RrKRbK4QlRHYEN1xXRZP3msS/qpbmug+s2q3Elc'
        'LrvIcbxTP0FLqhMG4t3rnIBOL9wGYp5oirK2W2JVqCtcJXAT6lu+obZUIp8CQjMb3F4HOruXBkBU/RVDoII9QhyU'
        'mGcxDqrFsDBkWPu7dZbtpQCvZiD+eQ3MSpY3wEHBDdJxEIbGYI7BYH6MuYckBnINvuMqJyqbDYDnGhaIaEW+roxf'
        '0Wta5lvay3XltTNbxLdCUwMKUG1DEOBoibquTCVXeF9qLrm+rkwmL/xys1kNBdUlsMgjrinYGQhzK4q4t7BtbfON'
        'edqjd4x5nuJfzEG5Fkwop0aYKyionwGYW9S/KkoMQ4o0GFhznBnohNzYnhKnQ8siMtd8Ba6oMbzBpBHHgCiYLUfk'
        'DyHi8YdzDz7OmAeqF1iwdFyFgpRzHHGYCt08Tu+1RHrWz060GFPirUaQlLIMZJZvPAtDqwZrs09JTYgBlyUmBaeu'
        'wYGMbBYzDb9c+sZRxLNc19lCU5UM6FSywC8zxeTAzNKUydTbcoZsSEAes3QNdM013x/6i+YWMfefGtb3BbhdWXww'
        'ELmKXkV49TV1hWhCUJ1O9KyRKo16ipULgqB0Q51vbNRTFdhgu5gqzDqlE+ipugKwVOtgwc/iqUoA7n0fO6dm/aqK'
        'BiB70dxu87bVMUNpzZgRWEmtjkRbSksWg8Kf1UHpdmUKnsgZ7gMRgjcKBlWtj5h7AV+jbmXDngDpC5zFb85MtRq5'
        'ihBAVVHc7y/MH8IsiKMiZFdoHM0NgyZCAz3OgFQIPycsAt82IW+QJcBIymjxsSX6Akvq0m03sdWGTGhI0NOloAXi'
        'BFzMRGYusPkpFJzKAszjNN113FNNoOGpvahfofqG2GZJ8EsYT6xkdWpjR0rdqaw7xTqpgAezIBx5U5anwdADtucx'
        'kktBGg8mp6Kye6eppJZQ6mQND2iO7CLn/5KB6s1bzVtFsHpncf/nFTYGNmRGnjd/U2RUSjLkhgZm6bmGhVHnsVds'
        'JpCqBuoarZosZQlcBtAiq97XPJ8IzwuWm3RF7kBBIBL7Kwu52CuGW6KtDrsuOavlZlVxEUwooh15pmCW4Bk5D7ec'
        'YDrBAIuJpRkja1JtKSThKKOc/zW2p1/dJiqwHWGQUre3pm4UoTIZxgSg6mgtQ+FILIKvfpYVcSKK/2nKP4Ejkk83'
        'CsRSNYBx9aa8pTodAnMZIxbJ8nKCaAFkdt121fbAH54ogW2m7pEJ3Hzh6RzAZrodO60njDcuyZPDqNJXMKsqQ1Ln'
        'yHynns4XsvPURKZC+5wFeI7rnHowNe0Jxbo2LTh1al8KYQMnIKDDiSQpEK6C7T/z01d2V4XxpODsn2DKVqRNykwb'
        '+RFFHfzCvWkLA6FmUYAJhOoRDTrgsYljRUeLH/IYm0+ByMsO+GA0gNIRw067TyUpl47Be+dFFWKgdrf7dBc+oEl0'
        'icdIXzMalT/GNj+xSyd1UXlVATgf+RFeL/XPveN44onDsJlFZ2Npa7vbui24OkKRVg7+ol0hOCwfZS7/ruQtSDLW'
        '2pj8aN12mxvGnAPmIJwsX0hwcC8VKKgk1chl5chL/l60bgNUcbTXTelUpwXTrNDuGOV3dFXAr/KA7z4eM6YssYMO'
        'BEwRxAykDiqpY75LAF0henR5rIbiDB5DHTDNgnoHRCGAhcHPunfvAN7gArSvl7DUG7Iw7Fq3HL6ZGIrSQRgPT7pW'
        'y2lBMf3wIhhI13zYbrw1z0zHEJPhnYGIxGk1q418UhJ3agTbA7od47hvV89NnWU5qAOL2trViaLNzrlj9EB9WwGP'
        'uQM9lTu37X4NAEb+bKSIKcKJherxGYj8KRPZ6SdBYj0ZC5liFzTbluk+Tiamgx+MPpOIPgbTBD/O2SAx9Y0s3mcv'
        'IjIiJOPJ2MVu6Awe7TlE4Kecs9SyXWBSdh7kxxb2ZxsY+JurJgX/fenCeMMQlFFBo4eXDixEKL03PyVatXsHHMCl'
        'OpU8Kug12pINeEgc84YsG6azAeEv1hp21BGXGpR1tybdRiRc3pNwj4LkPnxSIxuP8j8ZK6cSkMtA4lKml/EbjVXC'
        'C2Y+rWeBXbKYGlxW0YBLEmBI83TYCbhUOMNqCrzoDlS4bH2/2wMXfk5/Y1cZfRv4o26vr+XhAod61nESNvADQZvk'
        'YNv08CFP5zXpG3Q56X6DG8O6scQlEWCMi/PD5xIMG6wlsD25Ze5/+55pX4HmUZK57CIYe6SHMPsKyoe0ogoxAQuU'
        '4SUGrFzAhpTirqE/TUJW0LfPC0BPufce4BnZzf2FluxiyJLc2MzByRrMcn6QsAbXUgREZlDoTuv2LYeUgMTQlV8W'
        'qa7a14XjVVztt9fX3eZi47X2N6+7Kzp8rV2rwxd0+Vr7D6/Li/VwlT5HdQ6LpdgnKFdeb7nm6ttiZRWYsXSlRop2'
        '6AM9EdApbHHtwKqUOJ6aiBK8NEQGzhtDuTNR8JLUYvalthsF7XRkSYopTxPKhU03eeAHBUq+ZsFm3a+3V+puas0Z'
        'WBq7o2o7VR8Pg7GfvpQuLnyg/690saaDpQKmhvqUElOu16vIoWDaeWV9ep0+hun/strook4PAZKvQxPVOMIrNYpp'
        'qVpCs0G5iDBO3XQyaGOBdaEDSb2Fdf9vFNcfXGl906qB7id64CJ64Cym/jC37sN0h+Qb81xWMvQhBJp2bzUX4k3N'
        'V17h9h1vPlIASjlYDlu9D2mVAKRwmu5a0zH4KQQPhpkzcSgDfG+Y3/G8O3+VEwsFEUXXmE3GzXiQgumMiwyRXdTK'
        '8xoq0eJrfUNRKdvdz3FVEut6BVJQLvdzVpaLNrzFnFrMNeg5Qc8rkBmdo9Tuh4KTPwY2Qq/2hug7G7rF2i4q2GKF'
        'kC2YVfTutzYtzOdMkziCtZThOXZLTLiDiIvcN6b4yt8tzIXTXb5I6ISFyePdHBFtgBHp9aoEHhGBWFtDYTZsLxv3'
        'kWjNx91WWh8peKs1TAupALI+JsQ9FQBeUslq0w97YZyvgkeAm8L1qYcMek08vO1n+XKf0b/Ae9IepqsGDII3i/TN'
        'BpbjjUCLcxbYPDzumnRLEb0HvBuZz8FfpTtZRnF9ElbR+rq49Ns1X9u+i/8DAD9Mjn2ovLNuy15rsOONRh17p4J8'
        'TUG+s43/U5C/XiLPEsTRM/M4Mfs0wLMgCwbgOcjbAgpQirmuWrAqupCNJSBRYRW3KHWMgzjP4+kSSIkUr2OCDsZ7'
        'v/xYPjY7Ngt2iF/FvdruHfd1qBScQE4WsyrOVtEsllN7oZ4Z1W4v402BVrOpCsBFGEzxmJYFSpCnX0XazW6gpSqK'
        '6OTOzZYYLzbEcWQYStD8KfYKKEdz2+t02uXZEH/Qa7T6mN27wC8dAYXq1MMUu8PL+xptvAuorXQrTvz1MMvXmVya'
        '5Dad0fERf1CkGfAKvkiM4wmHqZ+CXZE7IHPCgW8j5PgmQjzLyS6CyWEpJl7FeSjO0XEwwcwu6oswd7PZAFFnFhTT'
        'BFm33buOcQuWANrgl2M/UezTkZfyXqNy1Ib2OInmnvw0AT4An8js4xYZ/OjTmU3vDO+2ZeDoc2Ggm9yK84auTYas'
        '4kO/9HBvB2QUtzfwAQRMLrIqTMY0CDXOEAzSvUvsBIZ2Y85x4ddyu49WApTy65o9pBwi98xjmMg28dIoJYigvGgC'
        's40HXLwBy88Z2DwQs3kD5ml+k0lNQH+XWqC1RodGCCsdKWu6zbuaFnGMJ3RwSN1mhn5wLgl/BXGpjsqbs0Q7noHE'
        '2+FdcRlYKyxunHeLy+i8orjELvGXxZy+lttSyW2DOFVG1HLvrpcDvnunGFAVcE2ojy4m6sVWyYKyKJcUwVrmPX4B'
        'guTH2F19x7C+ZZslGF8xFv8oi2kBWWIZJf6o+zqdY1MtjlzT/G65Je6Wd8WV8wj40YWhH4OxDZnY/2hLlT+bQnji'
        'DzEAoBcBOFQODiP2Re8NDIgJ2Bkvagt7gO8l5KjjvdCfwwK3eDXSh1XiSQWrXPqDQXzhBdHwmIFSppbmElgXAzEv'
        'm43HwYVluslobNrL2qPKGIYYYQGGQnnTmgblQe4vDPDEx3tPY3Bm4hjvXfA9BY8H/lk4m2gONO7y8mUvNFSGq04c'
        '3L7oFGo7y0fWaBSPQeTxeonYWEMn8sLmlw35d+NNo1WsGnG6iJQObiOMwWbHs2Qwt0oV5OjaBsQfVZFQmq4/mVjF'
        'EtvtWuYuNBgHaZYjm8TxFSgW3yqVsOp4HXzRq4gDwFysFd9VAMETqBbflMpyxcsDTKgGAVT+BlhSjBh3SpCMaQAZ'
        '0yrzkd4cCtSOlO032Ze6I1d2pwJSjxoYdaqBUL86qkrX2qae7Fzf+Su714GJgAookVABIyKqKCtklMdj8QUiSUel'
        'NGNDhZiFyiUNiKRqBdFUi54II7pszAKA3AbRiF0IwxbDECBI0mW9FPXlon2VwH1VQfsm5OibFJA/0KwvncSUcnqo'
        'PVdN+A6ReWaS8i5K+ZswWJriCy8noyAFk5BS7MkDEgamMffik64IhlD1Xw/Ht2zH6C4NszMLkAMF/rmHtynSUeZC'
        'oYl364HW4tKyUK9aG1HmDebCVNe2FMKqtRRlno+fE1bTjsfdIxf3qu+j/bWemmTIPXr9hXbylGMaSuISz2qYZQEA'
        'K7WXtkYHrxH3O5cRUeeVK7JvmId4uEQWkBMCcYj5VDWJlx1DfypkFuH58VEwHjM6LYmX1wO/eIZEnDNCRMFkFZCh'
        'RUUvNxx7U0/2xbdCiwignNx6mrXlZZj/ovxp4MkRn5ZGDeUVwCHE7gA7/2pjUIihYVxFeGX1GuY/PzC0gkWaNYiv'
        'Runj7FoC9bWPFB4UxzewxLBAHdSLRaYDfiVKEYNCK7tI5PWH8s2wG+RBLYR5WFjGdeKo3DGjg0VPC1Vp5iBioCcS'
        '5uewsLhyMdP4PDM7dP4d0YzGtARl755SLcuwPhJ4ZN1o3AM9iGmQSJwVsBUljVXelA4G6cA89ncEgH+xCECZAAAQ'
        'DiBxEFXIAhsARthQDUhhi0KP9GcBSH7VqYWheeA50wTyR7Z2/OGxwWvwlH9m+HR+V+RtA5io15ura02eZsa91nGQ'
        'Z5Wk3ure1qaBwnLMeLoWr6iKBnh2obgRl+HFCZCSM1bRPGKIGWk6TKKDmIVziu/xkuRc3ivflW8m8eHy83r0JJvi'
        'GOiGfcG9WjSJfQVnpbqGWZIUvKOeziJxWWSY4usaHbkmgAOrpMIbZYG4NCSuZfH7N3hKDO8wIWOgEjviPCuXIBhN'
        'YlvBxVENl/Jjv547HrGWBgQmUbpjHePpGfwHoeFNE10+D40f5rjlwuCxz6QHEH01ABJ3W+IQcw1fkeuXPH/jTBBX'
        'ndMo9vv439w47u9sPlxveltv7Wztuvj2pmm752mQMw+jWQtL3NFsmmQWKQrwMqIM94P8bBgEMlgOiHNdfFwQyItH'
        'GBebs3zcuCOU1JS/IaLoGKEggTN6UKlpjEKhOUuUkhrukdiKENMuD45pq18VTLlEoFm5+J3r1vzUY6czP+TuREWa'
        'pwbVZfiE5wDUeDwW+dJVERYaW66xJS+7ohiyC3+IEidfJyMoOqsa0OMyxkmDzjzj0sD7RnRtSJNJL5uClIGsVUjB'
        'R9NI7ul5NNd4oJsY3n08yEG5gPwP5mWfDUovlT1Hcg/QwJQHq3QPa8/Pa7pvb4ne17a+qc7l1mQSJAwzVNgxbqry'
        'XU/jpvHWo28bQ9Aw0EcMq2YDfSipmnWFC8Cocvn+DSpbUMo5NavoX7V/dKfo5A2uQlV1ipPQjfIk9DLvChay7sNA'
        'h8XpeTpExlnaX1i3ctXgyrp63SLEl1y2lGzEw5jlSQDzNUN3biqGp9aVAeUb0ZtmQT7vGLiSjfWmmInsjyKVqa+9'
        'ZuyA/5CCUoN2I5YFk0iDGJvF2dZOhRRXR9Uw9nkPHeyNeuVdwnywoo/MNb4uk/2qxnmB3u/yCqPVwWWLey/gHxpb'
        'uPrRFaiqFdc4PIYqIlncEeDvfYb0fnC52oCqUcy4CkELu6BfllGSdfjziw5XIw4nC1e3oyzvQ6AN8+ViXNcStEGj'
        'ieGvFPT0LMUGL68CqqSKBDHu4HUoQdw12t9yjFvwn+u6Dsz9txbaPJrlMEMwOmlvHePlF62jxx2cE+i0NITTUnH3'
        'XeP+gtvlaH4NeTHcfVnV/RTFR1HdIjAjeNExM7Lj+NxARhn/+38aBzuVgf5RBMtpanAjplWp+kEauWt0hIqWED8S'
        'CSoR3+sdyIQAf4/4FkXkuLcwis/xsaMy/F7EuymD7EHBLwi3NewyKnOT4CzOBT49+Vvk6btKzoyPtGtqWTDTvtFq'
        'Nm1BalsntYa+upC6pFWj8yuSWU2ymXY9P2uI1M3KN0LdYrruFeirhM7KVGMQ/XWSWZP5u4bOvrCzmJ+rRt+KJfP4'
        '8QZ3OtJtLwzRdB/HELKS/VxqXXkH6IFvv73jff7ej5791x+tN5//za/c/CLXUSrW7/kv/ubZj//kxSf//dn7//bT'
        'jz6uWuNPP/r3FWvz2Xu//uzjv372w//47C//5NkP//jF3733/Gd/9/xnf/r8B3/x/N2/hJLnv/yzF3/+gxe/+Ivn'
        'f/tf0FICBZyYZx/+8vNfv//itz95/stf/OO7/waGVU7jpx//9MVvf/DZh79/8bsPv/jkz8lK/OO7/5rMBH0+/0/v'
        'vfjb30NXW1988sMXH/zm+Y//24uf/hUvffa7nwAZX3zy/qcf/ejT3/0DlC+apC8++XeEp71FH2tbFQrG1Wym8dmv'
        '3ofunn38U2QLjOP//Nxor+Pf6xh6yKzEZXUkz//Hn3760a+RJz/68PMP/hgY8uyHf/3Z73/y7L2PC/MAJCirHn7p'
        'y+vZf3i/muJ59sHvP/vVX33xyQeVcqCNd/j5f/7Biw9/DlHvp7/98ecf/P2zT97F+fkQhvCbz/7hz2Asn370LswE'
        'wDz/Xz+D+Xn+87///Oe/4cSXa6MqWTwRTX/LQ3UweHplftW4SpbXKVjk5/GEkdlVRRKwmfLYhMTsUsY5s+xOWTSL'
        'QOhPRPa79rSsBMVcpemU1Q/2vO2d+w83D3e2Fw7TPhnz1WDRnQvPwxaeR7ePhyj3XRPcmrphHe5/Z2e96SZzU3/Z'
        'I6GLLbD+3HQSxgPLvGHaC29DJW6QUVc0QklCUvaKF2ESN2Whj+/1enlMS9rlafnKVeFy2EVGUdjIFXkmGYYgFrMl'
        'F/UJAzh5/pB/aBuuuPWLfcorWoouQKE3+U0ts2Nu8y+OWVxCgcL78P0AvzcO46TVbLRxR5RORYNBhPqtB/c39xvw'
        'XdSv75qXPYU0cRoPZErvbFTXWXn7JUdkXvtE70x+9VpNEQh7680mA98cP5t1PePI8VqsYMJLivj9/Udvewdb+/wR'
        'YY9nQArhphksBZv/syRuOs1Txmh67aLrml0YTbPLDR/7ZbdrtsQzT/qlP5p3KU1T73QGYSE9I4dJT/DJt1ZXb8nK'
        'Yz8cazVtWUOpCTzuIn6P4tmALom0b8gi8Hv5vZG1oogbUZrdqbfFs1q3KDMiyKCEiahoUwWS4FQSJTyZQgVURS+s'
        'SxqoZI2XcBIuxcIh3V5eBqR4mAah81fb2PIEFO30FIeawFSXV9mKtJKtviJEKfTy39m4iW2UQ8eJlzfxDCxl31k6'
        '5tljPIBZAB3xxyf4aUlHnF1dfpyYfygPrinnistrBOIBxjkd4+ReD7EEp7jlyLe26SVmcmnSG5Fti7RiupgAK1/J'
        'Up6EUO4VI+F4ZLR8+QrTEV5x15r/LO+LL734j2xQxqbe/y9wy0R8BqZZzzPQKqr9hwBAWvg7AHiGSUqh4mAua9mu'
        'tEQxvb7Z4j8cgP8aBRYREpLoa7GAtOt9c7m/tt1atR1fHUo7/YoGOcMOJtLJ3OgbH/Jx2uq7RuLIH85DEZR6+K/4'
        'iJMr4gxcOU+LFziuXRjlW1Foj/VYGdpRwo1y7gFEtNBHMA6G1XemeHsSN+U5M/5bDaCNWWLkMd7WjsK5I7cBGN8V'
        'yNS9ALfufSeVA3igUUx2/QtPypOI9Y8/qgtp+Rt216KTD1/gmqJJUZYWXrMH8wgf9EAg1trix27No1KVtwuXPq6I'
        'XZVs3tU7XHuFDvnVgS/xOF0N4/yEv+tVI2tGA8RwocmSZwJQmGvuEQ3xWsGrvC+0+GINvTp2xUMY2JxouUqt1lxi'
        'FGGOvIFXKz60FUrX8GizmuERCq58l++2AGHH8Yj+MRdV+ZjlLinUSfVQ36u6YsxORYUUR5A6irTQLk/hF2wtwatF'
        '+B2p3Mxd+L5LGPgpvQ7OqTyX1xG2kvE+5Gm8TuVNB1yX9pJuix04Sg4ubPvTTltHyqJj3LgxHS0gUm7laadbxDQW'
        'x1OEq1JzHqW8Kbl4mAA3WjLc1FBS2sZT+LzZulx9Wjg3l6i5n9YslQZ3Zjpue3xJhwxgxo61Q0O4YOvo/loOHIjj'
        'naMxvaohsV3/ckaFCzuSjDKyN3Cv1TEmEAk8FV1clv+IirhYWZ4ZkAcQYDb4mwFEkHLoqPQT7VcmjxzOEd/pwrR7'
        '03Vv3S2JUR/yxzCTNs/wqG3vK+4Bq+fjvwTZURw1OC2G3NHSswmL4WtxJ+DrODf80hEzBsz4eLclxljEgNyHXzVf'
        'MsfhxXwroIj/JKKlMWARcmv9vmyIJ64/P4x9TNsYPCyXOxyZ67o1C/I+6Jn5SFxZ9EbIO/jAg6Ev9XpLXb9F5H9d'
        '1zCH87Hsesy7HtPT6csftOATsBjvW2s25ZGqz1zUEVikHsRrodfRiZdnhpLOIadzKOmsuejNabwm57BIrWQnVFRe'
        'qeGZGn5hI6Rtq55VJEKKKaSZA+NkKakRp+AyMZdqi7yIUwyNRmQr//7GtW73E3AEPPg/skHJLemUKjmksiGMTvoa'
        'mON6ktiLZmkrxqchSAMDRtAdT58k3OZYdUYnb9qrt7nRwaC1Zg6FBGw+fOgdPdg7MMu38J+IiCbrCKAn9soKrFWP'
        '9IfnkaPueagPPE/46lw5rPxf/p2qWZp2AAA='
    ),
    'facescrub_kcenter': (
        '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'
    ),
    'facescrub_gradmatch': (
        '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'
        'd97yHHkAAA=='
    ),
    'facescrub_spot': (
        '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'
    ),
}

ORIGINAL_SCRIPT_SHA256 = {
    'cifar_kcenter': 'a54f3e8c46c920d731a3c2e625ebddcd21369836f58a92a1774fce452e9221b2',
    'cifar_gradmatch': '6492fd61fa699a3f58a34da583601082c6a6622c4d9e14f1193f97f4b0f23e7a',
    'cifar_spot': 'aaeabad3226cc27422ff7e1b3ace31500f035b23608565d8e9102b46d8e008ca',
    'digits_kcenter': 'bc4a46356cf61443e278a602ffcca26f3713fb849478a048df8106042f83c712',
    'digits_gradmatch': 'f3f4d8062b0bcdc61c3f8016786663487bc6b550809f69d7c6beb9d11f0a4ce8',
    'digits_spot': '55999e0a4a106e84e7ce0b37594bf8f0bd7e2830603657c38e237f0ce614509f',
    'self_m_all': 'f93555b0ea265de25985aa13d1c28d3f7bef9ca50add943fe171ecc5db76ebb0',
    'facescrub_kcenter': '208a1f1ea43bb55b2f8e7783023f895e536a0332af6b1024261c4651eec44f71',
    'facescrub_gradmatch': 'e92bac43b380dbfe23fceff64eccd2639f08c4b179d5fa4c1a2891d169d72a96',
    'facescrub_spot': 'ac9aef5deec1c7124c5646923cbd0631e4177f051d6259f77ded84576e8bd7fd',
}

@dataclass(frozen=True)
class ScriptFile:
    source_key: str
    filename: str

@dataclass(frozen=True)
class Experiment:
    key: str
    script_filename: str
    output_zips: tuple[str, ...]
    needs_cifar: bool = False
    needs_facescrub: bool = False

SCRIPT_FILES = [
    ScriptFile('cifar_kcenter', 'run_cifar100_kcenter_init_sensitivity_REAL50.py'),
    ScriptFile('cifar_kcenter', 'dependency_run_cifar100_kcenter_init_sensitivity_REAL50.py'),
    ScriptFile('cifar_gradmatch', 'run_cifar100_gradmatch_model_sensitivity_REAL50.py'),
    ScriptFile('cifar_spot', 'run_cifar100_spot_L_sensitivity_REAL50.py'),
    ScriptFile('digits_kcenter', 'run_digits_kcenter_init_sensitivity_REAL50.py'),
    ScriptFile('digits_gradmatch', 'run_digits_gradmatch_model_sensitivity_REAL50.py'),
    ScriptFile('digits_spot', 'run_digits_spot_L_sensitivity_REAL50.py'),
    ScriptFile('self_m_all', 'run_self_m_sensitivity_TRUE50.py'),
    ScriptFile('facescrub_kcenter', 'run_facescrub_kcenter_init_sensitivity_50.py'),
    ScriptFile('facescrub_gradmatch', 'run_facescrub_top10_gradmatch_model_wass_js_REAL50.py'),
    ScriptFile('facescrub_spot', 'run_facescrub_spot_L_sensitivity_REAL50.py'),
]

EXPERIMENTS = [
    Experiment('digits_kcenter', 'run_digits_kcenter_init_sensitivity_REAL50.py', ('digits_kcenter_(2)\u91cd\u590d50\u6b21.zip',)),
    Experiment('digits_gradmatch', 'run_digits_gradmatch_model_sensitivity_REAL50.py', ('digits_GRADMATCH\u91cd\u590d50\u6b21.zip',)),
    Experiment('digits_spot', 'run_digits_spot_L_sensitivity_REAL50.py', ('digits_SPOT\u91cd\u590d50\u6b21.zip',)),
    Experiment('self_m_all', 'run_self_m_sensitivity_TRUE50.py', ('digits_self_m_sensitivity_5groups_fromscratchK\u91cd\u590d50\u6b21.zip', 'facescrub_top10_2k_self_m_sensitivity_5groups_fromscratchK\u91cd\u590d50\u6b21.zip', 'cifar100_10classes_500each_5000_self_m_sensitivity_5groups_fromscratchK\u91cd\u590d50\u6b21.zip'), needs_cifar=True, needs_facescrub=True),
    Experiment('cifar_kcenter', 'run_cifar100_kcenter_init_sensitivity_REAL50.py', ('cifar100_kcenter\u91cd\u590d50\u6b21.zip',), needs_cifar=True),
    Experiment('cifar_gradmatch', 'run_cifar100_gradmatch_model_sensitivity_REAL50.py', ('cifar100_10classes_500each_5000_GRADMATCH\u91cd\u590d50\u6b21.zip',), needs_cifar=True),
    Experiment('cifar_spot', 'run_cifar100_spot_L_sensitivity_REAL50.py', ('cifar100_10classes_500each_5000_SPOT\u91cd\u590d50\u6b21.zip',), needs_cifar=True),
    Experiment('facescrub_kcenter', 'run_facescrub_kcenter_init_sensitivity_50.py', ('facescrub_kcenter\u91cd\u590d50\u6b21.zip',), needs_facescrub=True),
    Experiment('facescrub_gradmatch', 'run_facescrub_top10_gradmatch_model_wass_js_REAL50.py', ('facescrub_top10_2k-gradmatch\u91cd\u590d50\u6b21.zip',), needs_facescrub=True),
    Experiment('facescrub_spot', 'run_facescrub_spot_L_sensitivity_REAL50.py', ('facescrub_top10_2k_SPOT\u91cd\u590d50\u6b21.zip',), needs_facescrub=True),
]

OUTPUT_DIR_NAMES = [
    'digits_srot_vs_kcenter_init_wass_js_\u91cd\u590d50\u6b21',
    'digits_srot_vs_gradmatch_model_wass_js_REAL50',
    'digits_srot_vs_spot_L_wass_js_REAL50',
    'self_m_sensitivity_5groups_fromscratchK_outputs',
    'cifar100_10classes_500each_5000_srot_vs_kcenter_init_wass_js_REAL50',
    'cifar100_10classes_500each_5000_srot_vs_gradmatch_model_wass_js_REAL50',
    'cifar100_10classes_500each_5000_srot_vs_spot_L_wass_js_REAL50',
    'facescrub_top10_2k_srot_vs_kcenter_init_wass_js_50',
    'facescrub_top10_2k_srot_vs_gradmatch_model_wass_js_REAL50',
    'facescrub_top10_2k_srot_vs_spot_L_wass_js_REAL50',
    'digits_kcenter_REAL50_outputs',
]

ZIP_NAME_OVERRIDES = {
    'cifar100_10classes_500each_5000\u91cd\u590d50\u6b21.zip': 'cifar100_kcenter\u91cd\u590d50\u6b21.zip',
    'facescrub_top10_2k_srot_vs_kcenter_init_wass_js_50_results_code.zip': 'facescrub_kcenter\u91cd\u590d50\u6b21.zip',
    'facescrub_top10_2k_srot_vs_gradmatch_model_wass_js_REAL50_results_code.zip': 'facescrub_top10_2k-gradmatch\u91cd\u590d50\u6b21.zip',
}

EXTRA_ZIPS_TO_REMOVE = [
    'facescrub_top10_2k_srot_vs_gradmatch_model_wass_js_REAL50_distribution_only.zip',
]

SKIMAGE_FALLBACK_SOURCE = r'''
"""Minimal fallback for the resize/hog calls used by the Digits experiments."""
from __future__ import annotations

import numpy as np


def _resize_array(image, output_shape):
    arr = np.asarray(image, dtype=np.float64)
    output_shape = tuple(int(x) for x in output_shape)
    full_shape = output_shape + arr.shape[len(output_shape):]
    try:
        from scipy.ndimage import zoom
        factors = tuple(float(o) / float(i) for o, i in zip(full_shape, arr.shape))
        return zoom(arr, factors, order=1)
    except Exception:
        from PIL import Image
        mn = float(np.nanmin(arr))
        mx = float(np.nanmax(arr))
        scale = 255.0 / (mx - mn) if mx > mn else 1.0
        im = Image.fromarray(np.clip((arr - mn) * scale, 0, 255).astype(np.uint8))
        im = im.resize((int(output_shape[1]), int(output_shape[0])), Image.Resampling.BILINEAR)
        out = np.asarray(im, dtype=np.float64) / scale + mn
        return out


class transform:
    @staticmethod
    def resize(image, output_shape, anti_aliasing=True, preserve_range=True):
        return _resize_array(image, output_shape)


class color:
    @staticmethod
    def rgb2gray(image):
        arr = np.asarray(image, dtype=np.float64)
        if arr.ndim < 3 or arr.shape[-1] == 1:
            return np.squeeze(arr)
        return arr[..., :3] @ np.array([0.2125, 0.7154, 0.0721], dtype=np.float64)


def hog(image, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1),
        block_norm='L2-Hys', feature_vector=True):
    arr = np.asarray(image, dtype=np.float64)
    gy, gx = np.gradient(arr)
    magnitude = np.hypot(gx, gy)
    angle = (np.degrees(np.arctan2(gy, gx)) % 180.0)
    cell_h, cell_w = pixels_per_cell
    n_cells_y = arr.shape[0] // cell_h
    n_cells_x = arr.shape[1] // cell_w
    hist = np.zeros((n_cells_y, n_cells_x, int(orientations)), dtype=np.float64)
    bin_width = 180.0 / float(orientations)
    for cy in range(n_cells_y):
        ys = slice(cy * cell_h, (cy + 1) * cell_h)
        for cx in range(n_cells_x):
            xs = slice(cx * cell_w, (cx + 1) * cell_w)
            bins = np.floor(angle[ys, xs] / bin_width).astype(int) % int(orientations)
            weights = magnitude[ys, xs]
            for b in range(int(orientations)):
                hist[cy, cx, b] = float(weights[bins == b].sum())
    eps = 1e-12
    norm = np.sqrt(np.sum(hist * hist, axis=2, keepdims=True) + eps)
    hist = hist / norm
    if block_norm == 'L2-Hys':
        hist = np.minimum(hist, 0.2)
        norm = np.sqrt(np.sum(hist * hist, axis=2, keepdims=True) + eps)
        hist = hist / norm
    return hist.ravel() if feature_vector else hist
'''.lstrip()

FACESCRUB_ALIASES = {
    'facescrub_top10_2k.zip',
    'facescrub_top10_2k(1).zip',
    'facescrub_top10_2k(2).zip',
    'facescrub_top10_2k(3).zip',
}


def decode_payload(key: str) -> str:
    encoded = ''.join(SCRIPT_PAYLOADS[key])
    return gzip.decompress(base64.b64decode(encoded)).decode('utf-8')


def repo_default_dir() -> Path:
    return Path(__file__).resolve().parent


def unique_existing(paths: Iterable[Path]) -> list[Path]:
    seen = set()
    out = []
    for p in paths:
        rp = p.resolve() if p.exists() else p.absolute()
        key = os.path.normcase(str(rp))
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def find_dataset_zip(requested_dir: Path, filename: str) -> Path:
    script_dir = repo_default_dir()
    candidates = unique_existing([
        requested_dir / filename,
        script_dir / filename,
        Path.cwd() / filename,
        script_dir / '\u5b9e\u9a8c\u7ed3\u679c' / filename,
        script_dir.parent / '\u5b9e\u9a8c\u7ed3\u679c' / filename,
    ])
    for p in candidates:
        if p.exists():
            return p.resolve()
    tried = '\n  - '.join(str(p) for p in candidates)
    raise FileNotFoundError(f'Cannot find {filename}. Tried:\n  - {tried}')


def output_zip_names(experiments: list[Experiment]) -> list[str]:
    names = []
    for exp in experiments:
        names.extend(exp.output_zips)
    return names


def selected_experiments(keys: list[str] | None) -> list[Experiment]:
    if not keys:
        return EXPERIMENTS[:]
    wanted = set(keys)
    unknown = wanted.difference({e.key for e in EXPERIMENTS})
    if unknown:
        raise ValueError(f'Unknown experiment key(s): {sorted(unknown)}')
    return [e for e in EXPERIMENTS if e.key in wanted]


def script_files_for(experiments: list[Experiment]) -> list[ScriptFile]:
    filenames = {exp.script_filename for exp in experiments}
    if any(exp.key == 'cifar_spot' for exp in experiments):
        filenames.add('dependency_run_cifar100_kcenter_init_sensitivity_REAL50.py')
    return [spec for spec in SCRIPT_FILES if spec.filename in filenames]


def budget_ratios(max_budget_points: int | None) -> list[float]:
    if max_budget_points is None:
        return DEFAULT_BUDGET_RATIOS
    if max_budget_points < 1 or max_budget_points > len(DEFAULT_BUDGET_RATIOS):
        raise ValueError(f'--max-budget-points must be 1..{len(DEFAULT_BUDGET_RATIOS)}')
    return DEFAULT_BUDGET_RATIOS[:max_budget_points]


def path_literal(path: Path) -> str:
    return repr(str(path.resolve()))


def mapped_zip_name(name: str) -> str:
    return ZIP_NAME_OVERRIDES.get(name, name)


def patch_plot_labels(source: str) -> str:
    source = source.replace('Proposed-SROT', 'HiW-Core')
    source = source.replace('SROT', 'HiW-Core')

    source = re.sub(r"k-Center init=\{([^}]+)\}", r"k-Center ({\1} init)", source)
    source = re.sub(r"k-Center init=([A-Za-z0-9_.-]+)", lambda m: f"k-Center ({m.group(1)} init)", source)

    source = re.sub(r"SPOT L=\{([^}]+)\}", r"SPOT (candidate pool L={\1})", source)
    source = re.sub(r"SPOT L=([A-Za-z0-9_.+\-]+)", lambda m: f"SPOT (candidate pool L={m.group(1)})", source)

    source = source.replace('GradMatch zero-logit', 'GradMatch (zero-logit gradient)')
    source = re.sub(
        r"GradMatch logit C=([0-9.]+)",
        lambda m: f"GradMatch (logit C={m.group(1)} gradient)",
        source,
    )

    self_m_labels = {
        'HiW-Core m=C/4': 'HiW-Core (m=C/4)',
        'HiW-Core m=C/2': 'HiW-Core (m=C/2)',
        'HiW-Core m=C (label clusters)': 'HiW-Core (m=C label clusters)',
        'HiW-Core m=2C': 'HiW-Core (m=2C)',
        'HiW-Core m=3C': 'HiW-Core (m=3C)',
    }
    for old, new in self_m_labels.items():
        source = source.replace(old, new)
    return source


def patch_plot_style(source: str) -> str:
    source = re.sub(r"'font\.family'\s*:\s*'sans-serif'", "'font.family': 'serif'", source)
    source = re.sub(
        r"'font\.sans-serif'\s*:\s*\[[^\]]+\],",
        "'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'DejaVu Serif'],",
        source,
    )
    source = source.replace(
        "'font.size': 9.5, 'axes.labelsize': 10.5, 'axes.titlesize': 11.0, 'legend.fontsize': 8.0,",
        "'font.size': 10.5, 'axes.labelsize': 11.5, 'axes.titlesize': 12.0, 'legend.fontsize': 11.0, 'legend.title_fontsize': 11.5,",
    )
    source = re.sub(
        r"'font\.size'\s*:\s*9\.5,\s*'axes\.labelsize'\s*:\s*10\.5,\s*'axes\.titlesize'\s*:\s*11\.0,\s*'legend\.fontsize'\s*:\s*8\.0,",
        "'font.size': 10.5, 'axes.labelsize': 11.5, 'axes.titlesize': 12.0, 'legend.fontsize': 11.0, 'legend.title_fontsize': 11.5,",
        source,
    )
    source = source.replace("'lines.linewidth': 2.05", "'lines.linewidth': 2.8")
    source = source.replace("'lines.linewidth':2.05", "'lines.linewidth':2.8")
    source = source.replace(
        "linewidth=2.55 if is_prop else 1.75",
        "linewidth=3.65 if is_prop else 2.65",
    )
    source = source.replace(
        "linewidth=2.55 if is_eq else 1.95",
        "linewidth=3.65 if is_eq else 2.75",
    )
    source = source.replace(
        "markeredgewidth=1.1",
        "markeredgewidth=1.35",
    )
    source = re.sub(
        r"plt\.subplots\(figsize=\(\s*6\.9\s*,\s*4\.55\s*\)\)",
        "plt.subplots(figsize=(6.2, 6.2))",
        source,
    )
    source = re.sub(r"ax\.grid\(True,\s*which='major'[^)\n]*\)", "ax.grid(False)", source)
    source = re.sub(r"ax\.grid\(True,\s*which='minor'[^)\n]*\)", "ax.grid(False)", source)
    source = re.sub(
        r"(ax\.tick_params\(axis='both',\s*which='both',\s*labelsize=8\.7,\s*width=0\.85\))",
        r"\1; ax.set_box_aspect(1) if hasattr(ax, 'set_box_aspect') else None",
        source,
    )

    def replace_legend(match: re.Match[str]) -> str:
        indent = match.group(1)
        return f"{indent}# Legend omitted; use shared_legend_long.png for the standalone legend."

    source = re.sub(r"(?m)^(\s*)ax\.legend\([^\n]*\)\s*$", replace_legend, source)
    source = source.replace(", bbox_inches='tight'", "")
    source = source.replace(",bbox_inches='tight'", "")
    return source


def write_shared_legend(output_dir: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    legend_specs = [
        ('HiW-Core', '#6A3D9A', 'P', '-'),
        ('k-Center (mean init)', '#009E73', 'D', ':'),
        ('k-Center (random init)', '#0072B2', '^', '--'),
        ('k-Center (farthest_from_mean init)', '#D55E00', 'v', '-.'),
        ('k-Center (kmeans_center init)', '#E69F00', 's', '--'),
        ('k-Center (max_norm init)', '#666666', 'X', ':'),
        ('GradMatch (zero-logit gradient)', '#009E73', 'D', ':'),
        ('GradMatch (logit C=0.03 gradient)', '#0072B2', '^', '--'),
        ('GradMatch (logit C=0.1 gradient)', '#56B4E9', 'v', '-.'),
        ('GradMatch (logit C=1 gradient)', '#D55E00', 's', '--'),
        ('GradMatch (logit C=10 gradient)', '#E69F00', 'o', '-.'),
        ('GradMatch (logit C=30 gradient)', '#666666', 'X', ':'),
        ('SPOT (candidate pool L=K+30)', '#7F7F7F', 'o', '--'),
        ('SPOT (candidate pool L=K+60)', '#1F77B4', 's', '-.'),
        ('SPOT (candidate pool L=K+120)', '#D55E00', '^', ':'),
        ('SPOT (candidate pool L=K+240)', '#009E73', 'D', '--'),
        ('SPOT (candidate pool L=0.5N)', '#CC79A7', 'v', '-.'),
        ('SPOT (candidate pool L=full)', '#E69F00', 'X', ':'),
        ('HiW-Core (m=C/4)', '#999999', 'o', ':'),
        ('HiW-Core (m=C/2)', '#D55E00', 'D', '-.'),
        ('HiW-Core (m=C label clusters)', '#6A3D9A', 'P', '-'),
        ('HiW-Core (m=2C)', '#009E73', '^', '--'),
        ('HiW-Core (m=3C)', '#0072B2', 's', '--'),
    ]

    handles = [
        Line2D(
            [0], [0], color=color, linestyle=linestyle, marker=marker,
            markerfacecolor='white', markeredgecolor=color, markeredgewidth=1.55,
            linewidth=4.0, markersize=9.6,
        )
        for _, color, marker, linestyle in legend_specs
    ]
    labels = [label for label, _, _, _ in legend_specs]

    with plt.rc_context({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'DejaVu Serif'],
        'font.size': 16,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    }):
        fig = plt.figure(figsize=(34, 3.2), dpi=240)
        legend = fig.legend(
            handles, labels, loc='center', ncol=6, frameon=True, fancybox=False,
            framealpha=1.0, handlelength=1.6, columnspacing=1.1,
            handletextpad=0.45, borderpad=0.45, labelspacing=0.45,
            prop={'family': 'Times New Roman', 'weight': 'bold', 'size': 16},
        )
        legend.get_frame().set_linewidth(0.9)
        legend.get_frame().set_edgecolor('#222222')
        legend.get_frame().set_facecolor('white')
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / 'shared_legend_long.png', bbox_inches='tight', pad_inches=0.08)
        fig.savefig(output_dir / 'shared_legend_long.pdf', bbox_inches='tight', pad_inches=0.08)
        plt.close(fig)


def patch_optional_skimage_imports(source: str) -> str:
    source = source.replace(
        "from skimage import color, transform\nfrom skimage.feature import hog\n",
        "try:\n"
        "    from skimage import color, transform\n"
        "    from skimage.feature import hog\n"
        "except ModuleNotFoundError:\n"
        "    from _skimage_fallback import color, transform, hog\n",
    )
    source = source.replace(
        "from skimage import transform, color\nfrom skimage.feature import hog\n",
        "try:\n"
        "    from skimage import transform, color\n"
        "    from skimage.feature import hog\n"
        "except ModuleNotFoundError:\n"
        "    from _skimage_fallback import transform, color, hog\n",
    )
    source = source.replace(
        "from skimage import transform\nfrom skimage.feature import hog\n",
        "try:\n"
        "    from skimage import transform\n"
        "    from skimage.feature import hog\n"
        "except ModuleNotFoundError:\n"
        "    from _skimage_fallback import transform, hog\n",
    )
    return source


def patch_windows_path_literals(source: str) -> str:
    def repl(match: re.Match[str]) -> str:
        quote = match.group(1)
        path_text = match.group(2)
        if quote == "'":
            path_text = path_text.replace("'", "\\'")
        else:
            path_text = path_text.replace('"', '\\"')
        return f"Path(r{quote}{path_text}{quote})"

    return re.sub(r"Path\((['\"])([A-Za-z]:\\[^'\"]*)\1\)", repl, source)


def patch_source(source: str, *, work_dir: Path, output_dir: Path, cifar_zip: Path | None,
                 facescrub_zip: Path | None, repeats: int, ratios: list[float]) -> str:
    def map_mnt_suffix(suffix: str) -> Path:
        suffix = suffix.replace('\\', '/').lstrip('/')
        basename = Path(suffix).name
        if basename == CIFAR_ZIP_NAME:
            if cifar_zip is None:
                raise FileNotFoundError(CIFAR_ZIP_NAME)
            return cifar_zip
        if basename in FACESCRUB_ALIASES:
            if facescrub_zip is None:
                raise FileNotFoundError(FACESCRUB_ZIP_NAME)
            return facescrub_zip
        if basename.endswith('.py'):
            return work_dir / basename
        if basename.endswith('.zip'):
            return output_dir / mapped_zip_name(basename)
        return output_dir / suffix

    def repl_mnt(match: re.Match[str]) -> str:
        suffix = match.group(2) or ''
        return f"Path({path_literal(map_mnt_suffix(suffix))})"

    patched = re.sub(r"Path\((['\"])/mnt/data(?:/([^'\"]*))?\1\)", repl_mnt, source)

    def repl_parent_output(match: re.Match[str]) -> str:
        suffix = match.group(2)
        target = output_dir / mapped_zip_name(Path(suffix).name)
        return f"Path({path_literal(target)})"

    patched = re.sub(r"Path\(__file__\)\.resolve\(\)\.parent\s*/\s*(['\"])(facescrub_top10_2k_srot_vs_gradmatch_model_wass_js_REAL50(?:_results_code|_distribution_only)?(?:\.zip)?)\1", repl_parent_output, patched)

    if facescrub_zip is not None:
        patched = re.sub(r"BASE\s*/\s*(['\"])facescrub_top10_2k(?:\(\d\))?\.zip\1", f"Path({path_literal(facescrub_zip)})", patched)
        patched = re.sub(
            r"Path\(['\"][^'\"]*facescrub_top10_2k(?:\(\d\))?\.zip['\"]\)",
            f"Path({path_literal(facescrub_zip)})",
            patched,
        )
        patched = re.sub(
            r"read_facescrub_zip_features\(Path\([^)]*\)",
            f"read_facescrub_zip_features(Path({path_literal(facescrub_zip)})",
            patched,
        )
    if cifar_zip is not None:
        patched = re.sub(r"BASE\s*/\s*(['\"])cifar100_10classes_500each_5000\.zip\1", f"Path({path_literal(cifar_zip)})", patched)
        patched = re.sub(
            r"Path\(['\"][^'\"]*cifar100_10classes_500each_5000\.zip['\"]\)",
            f"Path({path_literal(cifar_zip)})",
            patched,
        )
        patched = re.sub(
            r"read_cifar_zip_features\(Path\([^)]*\)",
            f"read_cifar_zip_features(Path({path_literal(cifar_zip)})",
            patched,
        )

    patched = re.sub(r"^N_REPEATS\s*=\s*50\s*$", f"N_REPEATS = {int(repeats)}", patched, flags=re.MULTILINE)
    patched = re.sub(
        r"BUDGET_RATIOS\s*=\s*list\(np\.round\(np\.arange\(0\.02,\s*0\.51,\s*0\.02\),\s*2\)\)",
        f"BUDGET_RATIOS = {ratios!r}",
        patched,
    )
    patched = patch_plot_labels(patched)
    patched = patch_plot_style(patched)
    patched = patch_optional_skimage_imports(patched)
    patched = patched.replace('.to_markdown(', '.to_string(')
    patched = patch_windows_path_literals(patched)
    return patched


def clean_outputs(output_dir: Path, experiments: list[Experiment]) -> None:
    for name in OUTPUT_DIR_NAMES:
        target = output_dir / name
        if target.exists():
            shutil.rmtree(target)
    for name in output_zip_names(experiments):
        target = output_dir / name
        if target.exists():
            target.unlink()
    for name in EXTRA_ZIPS_TO_REMOVE:
        target = output_dir / name
        if target.exists():
            target.unlink()


def prepare_scripts(*, script_files: list[ScriptFile], work_dir: Path, output_dir: Path, cifar_zip: Path | None,
                    facescrub_zip: Path | None, repeats: int, ratios: list[float]) -> list[Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / '_skimage_fallback.py').write_text(SKIMAGE_FALLBACK_SOURCE, encoding='utf-8')
    written = []
    for spec in script_files:
        source = decode_payload(spec.source_key)
        patched = patch_source(source, work_dir=work_dir, output_dir=output_dir,
                               cifar_zip=cifar_zip, facescrub_zip=facescrub_zip,
                               repeats=repeats, ratios=ratios)
        target = work_dir / spec.filename
        with target.open('w', encoding='utf-8', newline='\n') as f:
            f.write(patched)
        written.append(target)
    metadata = {
        'original_script_sha256': ORIGINAL_SCRIPT_SHA256,
        'repeats': repeats,
        'budget_ratios': ratios,
        'output_dir': str(output_dir.resolve()),
        'cifar_zip': str(cifar_zip) if cifar_zip else None,
        'facescrub_zip': str(facescrub_zip) if facescrub_zip else None,
    }
    (work_dir / 'runner_patch_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    return written


def compile_scripts(paths: list[Path]) -> None:
    for path in paths:
        py_compile.compile(str(path), doraise=True)


def check_runtime_dependencies(python_exe: str) -> None:
    required = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'sklearn': 'scikit-learn',
        'PIL': 'Pillow',
        'matplotlib': 'matplotlib',
        'threadpoolctl': 'threadpoolctl',
    }
    code = (
        "import importlib, json; "
        "mods = " + repr(list(required)) + "; "
        "missing = []\n"
        "for m in mods:\n"
        "    try:\n"
        "        importlib.import_module(m)\n"
        "    except Exception as e:\n"
        "        missing.append([m, type(e).__name__, str(e)])\n"
        "print(json.dumps(missing))\n"
    )
    proc = subprocess.run([python_exe, '-c', code], text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f'Cannot run dependency check with {python_exe}: {proc.stderr.strip()}')
    missing = json.loads(proc.stdout or '[]')
    if missing:
        packages = ', '.join(required[m[0]] for m in missing)
        details = '; '.join(f'{m[0]}: {m[2]}' for m in missing)
        raise RuntimeError(
            f'Missing Python package(s) for experiment runtime in {python_exe}: {packages}. '
            f'Install with: "{python_exe}" -m pip install {packages}. Details: {details}'
        )


def run_experiment(exp: Experiment, *, work_dir: Path, output_dir: Path, timeout: int | None, python_exe: str) -> None:
    script = work_dir / exp.script_filename
    if not script.exists():
        raise FileNotFoundError(script)
    env = os.environ.copy()
    env.setdefault('PYTHONUTF8', '1')
    env.setdefault('PYTHONIOENCODING', 'utf-8')
    env.setdefault('OPENBLAS_NUM_THREADS', '1')
    env.setdefault('OMP_NUM_THREADS', '1')
    env.setdefault('MKL_NUM_THREADS', '1')
    print(f'=== Running {exp.key}: {script.name} ===', flush=True)
    start = time.perf_counter()
    proc = subprocess.run([python_exe, str(script)], cwd=str(output_dir), env=env, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f'Experiment {exp.key} failed with exit code {proc.returncode}')
    missing = [name for name in exp.output_zips if not (output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f'Experiment {exp.key} did not produce expected zip(s): {missing}')
    print(f'=== Finished {exp.key} in {(time.perf_counter() - start) / 60:.2f} min ===', flush=True)


def print_plan(*, data_dir: Path, output_dir: Path, cifar_zip: Path | None,
               facescrub_zip: Path | None, experiments: list[Experiment], repeats: int, ratios: list[float]) -> None:
    print('Plan')
    print(f'  data_dir: {data_dir.resolve()}')
    print(f'  output_dir: {output_dir.resolve()}')
    print(f'  cifar_zip: {cifar_zip if cifar_zip else "not required"}')
    print(f'  facescrub_zip: {facescrub_zip if facescrub_zip else "not required"}')
    print(f'  repeats: {repeats}')
    print(f'  budget_ratios: {ratios}')
    print('  experiments:')
    for exp in experiments:
        print(f'    - {exp.key} -> {", ".join(exp.output_zips)}')


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Unified runner for the 12 SROT experiment packages.')
    parser.add_argument('--data-dir', type=Path, default=repo_default_dir(), help='Directory containing CIFAR and FaceScrub dataset zip files. Default: this script directory.')
    parser.add_argument('--out-dir', type=Path, default=repo_default_dir(), help='Directory for result folders and zip packages. Default: this script directory.')
    parser.add_argument('--only', action='append', choices=[e.key for e in EXPERIMENTS], help='Run only one experiment key. Repeat for multiple keys.')
    parser.add_argument('--repeats', type=int, default=50, help='Number of true repeats. Default: 50.')
    parser.add_argument('--max-budget-points', type=int, default=None, help='For quick checks, use only the first N budget ratios. Default: all 25.')
    parser.add_argument('--timeout', type=int, default=None, help='Per-experiment subprocess timeout in seconds.')
    parser.add_argument('--python-exe', default=sys.executable, help='Python executable used for child experiment scripts. Default: the Python running this runner.')
    parser.add_argument('--skip-dependency-check', action='store_true', help='Skip runtime import checks before executing experiments.')
    parser.add_argument('--dry-run', action='store_true', help='Print the resolved plan without writing scripts or running experiments.')
    parser.add_argument('--prepare-only', action='store_true', help='Write patched embedded scripts and compile-check them, but do not run experiments.')
    parser.add_argument('--keep-work', action='store_true', help='Keep the temporary patched scripts under <out-dir>/_runner_work.')
    parser.add_argument('--no-clean', action='store_true', help='Do not remove previous known result folders/zips before running.')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.repeats < 1:
        raise ValueError('--repeats must be positive')
    args.data_dir = args.data_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    experiments = selected_experiments(args.only)
    ratios = budget_ratios(args.max_budget_points)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    needs_cifar = any(e.needs_cifar for e in experiments)
    needs_facescrub = any(e.needs_facescrub for e in experiments)
    cifar_zip = find_dataset_zip(args.data_dir, CIFAR_ZIP_NAME) if needs_cifar else None
    facescrub_zip = find_dataset_zip(args.data_dir, FACESCRUB_ZIP_NAME) if needs_facescrub else None

    print_plan(data_dir=args.data_dir, output_dir=args.out_dir, cifar_zip=cifar_zip,
               facescrub_zip=facescrub_zip, experiments=experiments,
               repeats=args.repeats, ratios=ratios)
    if args.dry_run:
        return 0

    work_dir = args.out_dir / '_runner_work' / 'patched_scripts'
    if work_dir.exists():
        shutil.rmtree(work_dir)
    if not args.no_clean:
        clean_outputs(args.out_dir, experiments)

    written = prepare_scripts(script_files=script_files_for(experiments),
                              work_dir=work_dir, output_dir=args.out_dir,
                              cifar_zip=cifar_zip, facescrub_zip=facescrub_zip,
                              repeats=args.repeats, ratios=ratios)
    compile_scripts(written)
    print(f'Prepared and compile-checked {len(written)} embedded scripts in {work_dir}')
    write_shared_legend(args.out_dir)
    print(f'Wrote standalone legend to {args.out_dir / "shared_legend_long.png"}')
    if args.prepare_only:
        return 0

    if not args.skip_dependency_check:
        check_runtime_dependencies(args.python_exe)

    for exp in experiments:
        run_experiment(exp, work_dir=work_dir, output_dir=args.out_dir, timeout=args.timeout, python_exe=args.python_exe)

    if not args.keep_work:
        shutil.rmtree(args.out_dir / '_runner_work', ignore_errors=True)
    for name in EXTRA_ZIPS_TO_REMOVE:
        extra = args.out_dir / name
        if extra.exists():
            extra.unlink()

    produced = [args.out_dir / name for name in output_zip_names(experiments)]
    print('Produced zip packages:')
    for path in produced:
        print(f'  {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
