"""tevador's MRL #146 simulator logic, one object at a time instead of per-second batches.

Everything is his (validation/sop_own_pending_20261117/tevador_sim.py: the
Blockchain class, share ids as slots with UNIQUE_SHARES, adopt_shares, the
round restarts, the stubbornness and success tests), except that each step
mines one object (the attacker's with probability alpha, a block with
probability 1/W) and the tests run after every object. His script mines a
whole second per step, honest miners first, and tests once per second. GAMMA
is 1 (ties go to the honest chain), COUNT_TIP_SHARES and UNIQUE_SHARES on,
UNCLE_BLOCKS off. Rate = successes per block interval (W objects).

    python3 -I tevador_events.py <alpha> <objects> <seed> [--depth 11] [--stubbornness 10]
"""
import random
import sys

W = 16
K = 3


class Blockchain:
    def __init__(self):
        self.reset()

    def reset(self):
        self.shares = set()
        self.num_blocks = 0
        self.weight = 0
        self.skip_weight = 0

    def soft_reset(self):
        self.num_blocks = 0
        self.weight = 0
        self.skip_weight = 0

    def add_share(self, shr):
        if len(self.shares) >= 200:
            return
        self.shares.add(shr)

    def adopt_shares(self, shares):
        self.shares = set()
        for shr in shares:
            self.add_share(shr)

    def total_work(self):
        return self.weight + len(self.shares)

    def score(self):
        return self.total_work()

    def block(self):
        wt = 1 + len(self.shares)
        self.num_blocks += 1
        self.weight += wt
        if self.num_blocks > 1:
            self.skip_weight += wt
        self.shares.clear()

    def share(self, other):
        shr = len(self.shares)
        self.add_share(shr)
        if other is not None and self.num_blocks == 0 and other.num_blocks == 0:
            other.add_share(shr)


def main():
    alpha, objects, seed = float(sys.argv[1]), int(float(sys.argv[2])), int(sys.argv[3])
    depth = int(sys.argv[sys.argv.index("--depth") + 1]) if "--depth" in sys.argv else 11
    stub = int(sys.argv[sys.argv.index("--stubbornness") + 1]) if "--stubbornness" in sys.argv else 10
    rnd = random.Random(seed)
    honest, attack = Blockchain(), Blockchain()
    n = attacks = 0
    reorg = False
    while n < objects:
        if reorg:
            honest.reset()
            attack.soft_reset()
            honest.adopt_shares(attack.shares)
        else:
            honest.soft_reset()
            attack.reset()
            attack.adopt_shares(honest.shares)
        reorg = False
        while honest.total_work() - attack.total_work() <= stub and n < objects:
            u = rnd.random()
            is_att = u < alpha
            is_blk = rnd.random() < 1.0 / W
            if is_att:
                attack.block() if is_blk else attack.share(None)
            else:
                honest.block() if is_blk else honest.share(attack)
            n += 1
            if honest.score() - attack.score() <= 0 \
                    and (attack.skip_weight >= W * (K - 1) or honest.skip_weight >= W * (K - 1)) \
                    and honest.num_blocks >= depth:
                break
        if n >= objects:
            break
        diff = honest.score() - attack.score()
        if attack.skip_weight < W * (K - 1) and honest.skip_weight < W * (K - 1) or honest.num_blocks < depth:
            continue
        if diff < 0:
            attacks += 1
            reorg = True
    print(f"alpha {alpha} seed {seed} objects {n} intervals {n / W:.6g} attacks {attacks}")


if __name__ == "__main__":
    main()
