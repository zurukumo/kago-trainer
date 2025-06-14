import base64
import hashlib
import struct

from mt19937ar import MT19937ar


class YamaGenerator:
    mt: MT19937ar

    __slots__ = ("mt",)

    def __init__(self, b64seed: str) -> None:
        N_INIT = 624

        seed = base64.b64decode(b64seed)
        init = list(struct.unpack(f"<{N_INIT}I", seed))

        mt = MT19937ar()
        mt.init_by_array(init)
        self.mt = mt

    def generate(self) -> list[int]:
        N_SRC = 288
        src: list[int] = []
        for i in range(N_SRC):
            src.append(self.mt.genrand_int32())

        N_RND = 136  # Actually 144
        rnd: list[int] = []
        for i in range(N_SRC // 32):
            s = struct.pack("<32I", *src[32 * i : 32 * (i + 1)])
            m = hashlib.sha512(s).digest()
            rnd.extend(struct.unpack("<16I", m))

        yama = list(range(N_RND))
        for i in range(N_RND):
            j = rnd[i] % (N_RND - i) + i
            yama[i], yama[j] = yama[j], yama[i]

        return yama
