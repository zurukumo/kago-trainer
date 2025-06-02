from kago_utils.actions import Ankan, Chii, Daiminkan, Kakan, Pon
from kago_utils.hai import Hai
from kago_utils.hai_group import HaiGroup
from kago_utils.zaichi import Zaichi


class HuuroParser:
    @classmethod
    def from_haihu(cls, m: int) -> Chii | Pon | Kakan | Daiminkan | Ankan:
        if cls.is_chii(m):
            return cls.parse_chii(m)
        elif cls.is_pon(m):
            return cls.parse_pon(m)
        elif cls.is_kakan(m):
            return cls.parse_kakan(m)
        elif cls.is_daiminkan(m):
            return cls.parse_daiminkan(m)
        elif cls.is_ankan(m):
            return cls.parse_ankan(m)

        raise ValueError("Invalid Huuro")

    @classmethod
    def is_chii(cls, m: int) -> bool:
        return bool(m & 0x0004)

    @classmethod
    def is_pon(cls, m: int) -> bool:
        return not bool(m & 0x0004) and bool(m & 0x0008)

    @classmethod
    def is_kakan(cls, m: int) -> bool:
        return not bool(m & 0x0004) and bool(m & 0x0010)

    @classmethod
    def is_daiminkan(cls, m: int) -> bool:
        return not bool(m & 0x003C) and cls.parse_from_who(m) != Zaichi.JICHA

    @classmethod
    def is_ankan(cls, m: int) -> bool:
        return not bool(m & 0x003C) and cls.parse_from_who(m) == Zaichi.JICHA

    @classmethod
    def parse_chii(cls, m: int) -> Chii:
        pattern = (m & 0xFC00) >> 10
        h1 = ((pattern // 3 // 7) * 9 + (pattern // 3 % 7 + 0)) * 4 + ((m & 0x0018) >> 3)
        h2 = ((pattern // 3 // 7) * 9 + (pattern // 3 % 7 + 1)) * 4 + ((m & 0x0060) >> 5)
        h3 = ((pattern // 3 // 7) * 9 + (pattern // 3 % 7 + 2)) * 4 + ((m & 0x0180) >> 7)
        stolen_hai = [h1, h2, h3][pattern % 3]
        return Chii(
            hais=HaiGroup.from_list([h1, h2, h3]),
            stolen=Hai(stolen_hai),
        )

    @classmethod
    def parse_pon(cls, m: int) -> Pon:
        pattern = (m & 0xFE00) >> 9
        base_id = (pattern // 3) * 4
        unused = Hai(base_id + ((m & 0x0060) >> 5))
        hais = HaiGroup.from_list([base_id, base_id + 1, base_id + 2, base_id + 3]) - unused
        stolen = hais[pattern % 3]
        from_who = cls.parse_from_who(m)
        return Pon(hais=hais, stolen=stolen, from_who=from_who)

    @classmethod
    def parse_kakan(cls, m: int) -> Kakan:
        return cls.parse_pon(m).to_kakan()

    @classmethod
    def parse_daiminkan(cls, m: int) -> Daiminkan:
        stolen_hai = (m & 0xFF00) >> 8
        h1 = stolen_hai - stolen_hai % 4
        from_who = cls.parse_from_who(m)
        return Daiminkan(
            hais=HaiGroup.from_list([h1, h1 + 1, h1 + 2, h1 + 3]),
            stolen=Hai(stolen_hai),
            from_who=from_who,
        )

    @classmethod
    def parse_ankan(cls, m: int) -> Ankan:
        h1 = ((m & 0xFF00) >> 8) // 4 * 4
        return Ankan(hais=HaiGroup.from_list([h1, h1 + 1, h1 + 2, h1 + 3]))

    @classmethod
    def parse_from_who(cls, m: int) -> Zaichi:
        return [Zaichi.JICHA, Zaichi.SIMOCHA, Zaichi.TOIMEN, Zaichi.KAMICHA][m & 0x0003]
