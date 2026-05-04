
import sys
import time
from z3 import (
    BitVec, BitVecVal, BitVecRef, Solver, sat, RotateLeft,
    Extract, Concat, ZeroExt, SignExt, LShR, If, Or
)


GOLDEN_RATIO_64 = -7046029254386353131
SILVER_RATIO_64 = 7640891576956012809

STAFFORD13_MUL1 = -4658895280553007687
STAFFORD13_MUL2 = -7723592293110705685

def to_u64(v):
    return v & 0xFFFFFFFFFFFFFFFF

GOLDEN_U = to_u64(GOLDEN_RATIO_64)
SILVER_U = to_u64(SILVER_RATIO_64)
MUL1_U = to_u64(STAFFORD13_MUL1)
MUL2_U = to_u64(STAFFORD13_MUL2)


def bv64(val):
    return BitVecVal(val & 0xFFFFFFFFFFFFFFFF, 64)


def mix_stafford13(z):
    z = (z ^ LShR(z, 30)) * bv64(MUL1_U)
    z = (z ^ LShR(z, 27)) * bv64(MUL2_U)
    z = z ^ LShR(z, 31)
    return z


def upgrade_seed_to_128bit(seed):
    lo_unmixed = seed ^ bv64(SILVER_U)
    hi_unmixed = lo_unmixed + bv64(GOLDEN_U)
    lo = mix_stafford13(lo_unmixed)
    hi = mix_stafford13(hi_unmixed)

    return lo, hi


def xoroshiro_next(s0, s1):
    result = RotateLeft(s0 + s1, 17) + s0
    s1_xor = s1 ^ s0
    new_s0 = RotateLeft(s0, 49) ^ s1_xor ^ (s1_xor << 21)
    new_s1 = RotateLeft(s1_xor, 28)
    return result, new_s0, new_s1


def worldgen_next_long(s0, s1):
    r0, s0, s1 = xoroshiro_next(s0, s1)
    upper_32 = Extract(63, 32, r0)

    r1, s0, s1 = xoroshiro_next(s0, s1)
    lower_32 = Extract(63, 32, r1)

    upper_64 = Concat(upper_32, BitVecVal(0, 32))
    lower_64 = SignExt(32, lower_32)

    result = upper_64 + lower_64

    return result, s0, s1


def handle_zero_state(s0, s1):
    is_zero = (s0 | s1) == bv64(0)
    new_s0 = If(is_zero, bv64(GOLDEN_U), s0)
    new_s1 = If(is_zero, bv64(SILVER_U), s1)
    return new_s0, new_s1

def py_mix_stafford13(z):
    z &= 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * MUL1_U) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * MUL2_U) & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF
    return z


def py_upgrade_seed(seed):
    seed &= 0xFFFFFFFFFFFFFFFF
    lo_unmixed = (seed ^ SILVER_U) & 0xFFFFFFFFFFFFFFFF
    hi_unmixed = (lo_unmixed + GOLDEN_U) & 0xFFFFFFFFFFFFFFFF
    lo = py_mix_stafford13(lo_unmixed)
    hi = py_mix_stafford13(hi_unmixed)
    if (lo | hi) == 0:
        lo = GOLDEN_U
        hi = SILVER_U
    return lo, hi


def py_rotl64(x, k):
    x &= 0xFFFFFFFFFFFFFFFF
    return ((x << k) | (x >> (64 - k))) & 0xFFFFFFFFFFFFFFFF


def py_xoroshiro_next(s0, s1):
    result = (py_rotl64((s0 + s1) & 0xFFFFFFFFFFFFFFFF, 17) + s0) & 0xFFFFFFFFFFFFFFFF
    s1 ^= s0
    new_s0 = (py_rotl64(s0, 49) ^ s1 ^ ((s1 << 21) & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
    new_s1 = py_rotl64(s1, 28)
    return result, new_s0, new_s1


def py_worldgen_next_long(s0, s1):
    r0, s0, s1 = py_xoroshiro_next(s0, s1)
    upper = (r0 >> 32) & 0xFFFFFFFF

    r1, s0, s1 = py_xoroshiro_next(s0, s1)
    lower = (r1 >> 32) & 0xFFFFFFFF

    if lower & 0x80000000:
        lower_ext = lower | 0xFFFFFFFF00000000
    else:
        lower_ext = lower

    result = ((upper << 32) + lower_ext) & 0xFFFFFFFFFFFFFFFF
    return result, s0, s1


def py_compute_ab(world_seed):
    s0, s1 = py_upgrade_seed(world_seed)
    a, s0, s1 = py_worldgen_next_long(s0, s1)
    a = a | 1
    b, s0, s1 = py_worldgen_next_long(s0, s1)
    b = b | 1
    return a, b


def py_to_signed64(v):
    v &= 0xFFFFFFFFFFFFFFFF
    if v >= (1 << 63):
        return v - (1 << 64)
    return v


def py_verify_seed(world_seed):
    a, b = py_compute_ab(world_seed)
    mask = (1 << 60) - 1

    sum_ab = (a + b) & 0xFFFFFFFFFFFFFFFF
    diff_ab = (a - b) & 0xFFFFFFFFFFFFFFFF

    if (sum_ab & mask) == 0:
        return True, "(1, 1)", a, b
    elif (diff_ab & mask) == 0:
        return True, "(-1, 1)", a, b
    else:
        return False, None, a, b


def find_diagonal_seeds_z3(mode="both", max_solutions=100, timeout_sec=3600, threads=4):
    print(f"=== Diagonal Seed Finder for MC 26.1 (Z3 SMT) ===")
    print(f"Mode: {mode}, Max solutions: {max_solutions}, Timeout: {timeout_sec}s, Threads: {threads}")
    print()

    world_seed = BitVec('world_seed', 64)
    s0_init, s1_init = upgrade_seed_to_128bit(world_seed)

    s0_init, s1_init = handle_zero_state(s0_init, s1_init)

    a_raw, s0_2, s1_2 = worldgen_next_long(s0_init, s1_init)
    a = a_raw | bv64(1)

    b_raw, s0_3, s1_3 = worldgen_next_long(s0_2, s1_2)
    b = b_raw | bv64(1)

    mask60 = bv64((1 << 60) - 1)

    solver = Solver()
    solver.set("timeout", timeout_sec * 1000)
    solver.set("threads", threads)

    if mode == "sum":
        solver.add((a + b) & mask60 == bv64(0))
    elif mode == "diff":
        solver.add((a - b) & mask60 == bv64(0))
    elif mode == "both":
        solver.add(Or((a + b) & mask60 == bv64(0), (a - b) & mask60 == bv64(0)))

    solutions = []
    start_time = time.time()

    result_file = open("result.txt", "a")
    print("Results will be saved to: result.txt")

    print("Searching...")
    for i in range(max_solutions):
        elapsed = time.time() - start_time
        if elapsed > timeout_sec:
            print(f"\nTimeout reached after {elapsed:.1f}s")
            break

        result = solver.check()
        if result == sat:
            model = solver.model()
            seed_expr = model.eval(world_seed)
            seed_val = seed_expr.as_long()  # type: ignore[union-attr]
            solutions.append(seed_val)

            valid, direction, a_val, b_val = py_verify_seed(seed_val)

            signed_seed = py_to_signed64(seed_val)
            elapsed = time.time() - start_time
            status = "VALID" if valid else "INVALID"
            print(f"  [{i+1}] seed={signed_seed} (0x{seed_val:016X}) "
                  f"dir={direction} [{status}] ({elapsed:.1f}s)")

            if not valid:
                print(f"       WARNING: Verification failed!")
                print(f"       a=0x{a_val:016X}, b=0x{b_val:016X}")
                print(f"       a+b=0x{(a_val+b_val)&0xFFFFFFFFFFFFFFFF:016X}")
                print(f"       a-b=0x{(a_val-b_val)&0xFFFFFFFFFFFFFFFF:016X}")

            if valid:
                result_file.write(
                    f"seed_signed={signed_seed} seed_unsigned={seed_val} "
                    f"dir={direction} a=0x{a_val:016X} b=0x{b_val:016X} mode=z3\n"
                )
                result_file.flush()

            solver.add(world_seed != BitVecVal(seed_val, 64))
        else:
            elapsed = time.time() - start_time
            if i == 0:
                print(f"  No solution found (result: {result}) after {elapsed:.1f}s")
            else:
                print(f"\n  No more solutions (result: {result}) after {elapsed:.1f}s")
            break

    result_file.close()
    elapsed = time.time() - start_time
    print(f"\n=== Found {len(solutions)} seed(s) in {elapsed:.1f}s ===")

    return solutions

if __name__ == "__main__":
    mode = "both"
    max_sol = 100
    timeout = 7200
    threads = 4

    if len(sys.argv) > 1:
        mode = sys.argv[1]
    if len(sys.argv) > 2:
        max_sol = int(sys.argv[2])
    if len(sys.argv) > 3:
        timeout = int(sys.argv[3])
    if len(sys.argv) > 4:
        threads = int(sys.argv[4])

    seeds = find_diagonal_seeds_z3(mode=mode, max_solutions=max_sol, timeout_sec=timeout, threads=threads)

    if seeds:
        print("\nAll found seeds (signed):")
        for s in seeds:
            signed = py_to_signed64(s)
            valid, direction, a_val, b_val = py_verify_seed(s)
            print(f"  {signed}  direction={direction}")
