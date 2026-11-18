"""features5.py — 第5版エンジン用。rate() が内訳（comp・mu_t・mkf）を自分で返すので、それをそのまま書き出す"""
import sys
import features as F

def make_native(K):
    class LM(K.Model):
        def rate(self, h, race, jk):
            out = super().rate(h, race, jk)
            # features.py が使う追加の項目
            rs = sorted((r for r in self.by_horse.get(h["name"], []) if r["ymd"] < race["date"]), key=lambda r: r["ymd"], reverse=True)
            fs = [r["F"] for r in rs[:12] if r["F"] is not None and (r["surf"] == "障") == (race["surf"] == "障")]
            out = dict(out, rs=rs, prior=self.clv.get(race["cls"], 0.0) - self.P["prior_gap"], top=None,
                       f_sd=(sum((f - sum(fs[:8]) / len(fs[:8])) ** 2 for f in fs[:8]) / len(fs[:8])) ** 0.5 if len(fs[:8]) >= 3 else None,
                       f_last=next((r["F"] for r in rs if r["F"] is not None and r["surf"] != "障"), None))
            self._log.append((h["name"], out))
            return out
    return LM

F.make_fmodel = make_native
_orig_work = F._work
F.COMP_KEYS = F.COMP_KEYS
def _work(group):
    rows = _orig_work(group)
    return rows
if __name__ == "__main__":
    F.main()
