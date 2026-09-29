// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if !defined(ARCH_BLACKHOLE)
#error "eth_clock_stream.hpp is Blackhole only"
#endif

#include <cstdint>

namespace sampler {
// sampler_stream loads these words in this order.
struct StreamArgs {
    volatile uint32_t* refclk;
    volatile uint32_t* wall;
    uint32_t period, pos8s;
    volatile uint32_t* slots;
    uint32_t mask;
    volatile uint32_t* head;
    uint32_t limit, iters;
    const uint32_t* pads;
    volatile uint32_t* tail_word;
    uint32_t tail;
};
static_assert(sizeof(StreamArgs) == 12 * sizeof(uint32_t));
}  // namespace sampler

/**
 * @brief Streams refclk updates into args->slots from args->tail on and returns the tail after the last one it stored.
 *
 * Update n goes to slot n & args->mask. Each pass first runs args->pads[tail & 255] / 4 nops, then reads blocks of
 * refclk, refclk, wall, refclk until the refclk steps. A update whose two blocks each took args->period wall ticks
 * is stored as its new refclk and the wall time of its step in eighths, the block's wall read plus its gap's signed
 * byte in args->pos8s; one that did not is dropped. With period 0 every update is stored, unpublished, as its block's
 * wall length, for calibration. Otherwise, before each stored update the tail it follows is written to
 * *args->tail_word, so every published sample was stored a handler earlier.
 *
 * At args->limit the stream reloads *args->head, the model's position, and moves the limit args->mask past it,
 * waiting there while the ring is full; the pass after a reload skips its pad, which pays for the reload. It returns
 * when the head is kSyncHeadStop past the model's position, or once args->iters passes have stored nothing (0 counts
 * 2^32).
 */
extern "C" uint32_t sampler_stream(sampler::StreamArgs* args);

// Registers: s0 and s1 hold the refclk and wall-clock register addresses, a0 the address of .Lprime, a1 the period (0
// while calibrating), a2 the gap positions, a3 and a4 the slots and their mask, a5 and a6 the head's address and the
// limit, a7 the passes left that may store nothing, t3 the tail word, t4 the pads, t5 the next pass's pad in bytes and
// t6 the tail. A pass jumps t5 bytes back from .Lprime into a run of nops, which is its pad. The block registers rotate
// so that a update's handler finds three blocks in place: x24/x25 hold the first refclk read of a block, x26/x27 the
// second, x18..x20 the wall read and x21..x23 the last refclk read.
asm(R"ASM(
    .section .text.sampler_stream, "ax", @progbits
    .option push
    .option norvc
    .altmacro

.macro sampler_block pos
    sampler_block_in %(24 + \pos % 2), %(26 + \pos % 2), %(18 + \pos % 3), %(21 + \pos % 3)
.endm
.macro sampler_block_in r1, r2, w, r3
    lw x\r1, 0(s0)
    lw x\r2, 0(s0)
    lw x\w, 0(s1)
    lw x\r3, 0(s0)
.endm

.macro sampler_step pos
    sampler_block \pos
    sampler_step_in \pos, %(21 + (\pos + 2) % 3), %(21 + (\pos + 1) % 3)
.endm
.macro sampler_step_in pos, r3_last, r3_before
    bne x\r3_last, x\r3_before, .Lstub\pos
.endm

.macro sampler_stub pos
    sampler_stub_in \pos, %(18 + \pos % 3), %(18 + (\pos + 2) % 3), %(18 + (\pos + 1) % 3), %(21 + (\pos + 2) % 3), %(21 + (\pos + 1) % 3), %(24 + (\pos + 1) % 2), %(26 + (\pos + 1) % 2)
.endm
.macro sampler_stub_in pos, w_now, w_last, w_before, r3_last, r3_before, r1_last, r2_last
    .balign 64
.Lstub\pos:
    # Both blocks took exactly one period, so no read in them was held up.
    sub t0, x\w_last, x\w_before
    sub t1, x\w_now, x\w_last
    beqz a1, .Lcal
    xor t0, t0, a1
    xor t1, t1, a1
    or t0, t0, t1
    bnez t0, .Lrej
    # Which of the block's first two reads still saw the old refclk names the gap the step fell in.
    xor t1, x\r1_last, x\r3_before
    seqz t1, t1
    xor t2, x\r2_last, x\r1_last
    seqz t2, t2
    sll t1, t1, t2
    slli t1, t1, 3
    srl t1, a2, t1
    sext.b t1, t1
    slli t2, x\w_last, 3
    add t2, t2, t1
    and t0, t6, a4
    sh3add t0, t0, a3
    sw x\r3_last, 0(t0)
    sw t2, 4(t0)
    sw t6, 0(t3)
    addi t6, t6, 1
    beq t6, a6, .Lrefill
    sub t1, a0, t5
    jr t1
.endm

    .balign 64
    .globl sampler_stream
sampler_stream:
    addi sp, sp, -48
    .irp r, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
    sw s\r, (4 * \r)(sp)
    .endr
    lw s0, 0(a0)
    lw s1, 4(a0)
    lw a1, 8(a0)
    lw a2, 12(a0)
    lw a3, 16(a0)
    lw a4, 20(a0)
    lw a5, 24(a0)
    lw a6, 28(a0)
    lw a7, 32(a0)
    lw t4, 36(a0)
    lw t3, 40(a0)
    lw t6, 44(a0)
    la a0, .Lprime
    csrrsi zero, 0x7c0, 2
    j .Lprime
    .balign 64
.Lpass:
    sub t1, a0, t5
    jr t1
    .rept 5
    nop
    .endr
.Lprime:
    andi t0, t6, 255
    sh2add t0, t0, t4
    lw t5, 0(t0)
    sampler_block 0
    sampler_block 1
    .irp pos, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
    sampler_step \pos
    .endr
.Lrej:
    addi a7, a7, -1
    bnez a7, .Lpass
.Lexit:
    csrrci zero, 0x7c0, 2
    mv a0, t6
    .irp r, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
    lw s\r, (4 * \r)(sp)
    .endr
    addi sp, sp, 48
    ret
.Lrefill:
    lw t0, 0(a5)
    sub t1, t6, t0
    bltz t1, .Lexit
    add a6, t0, a4
    beq a6, t6, .Lrefill
    jr a0
.Lcal:
    and t0, t6, a4
    sh3add t0, t0, a3
    sw t1, 0(t0)
    addi t6, t6, 1
    beq t6, a6, .Lrefill
    j .Lpass
    .irp pos, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
    sampler_stub \pos
    .endr

    .noaltmacro
    .option pop
    .text
)ASM");
