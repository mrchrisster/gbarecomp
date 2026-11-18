#include "bios_hle.h"
#include "gba_bus.h"
#include "gba_ppu.h"
#include "runtime_arm.h"
#include "runtime_bus_bridge.h"

#include <cstdint>
#include <cstdio>
#include <memory>

extern "C" uint64_t g_runtime_cycles;
extern "C" void runtime_mmio_catch_up(void);

namespace {

constexpr uint32_t kIrqHandlerAddr = 0x03000100u;
constexpr uint32_t kVCountOutAddr  = 0x03000200u;
constexpr uint32_t kIrqCountAddr   = 0x03000204u;
constexpr uint32_t kSwiReturnPc    = 0x03000022u;

void install_iwram_vblank_irq_handler(gba::GbaBus& bus) {
    // Runs in ARM mode entered from bios_dispatch_stub (R0 = 0x04000000, LR = 0x138).
    // Records VCOUNT at IRQ entry, acknowledges IF, sets INTR_CHECK VBlank bit, and
    // increments an IRQ counter.
    static constexpr uint32_t kCode[] = {
        0xE1D010B6u,  // ldrh r1, [r0, #6]      ; REG_VCOUNT
        0xE3A02403u,  // mov  r2, #0x03000000
        0xE5821200u,  // str  r1, [r2, #0x200]  ; kVCountOutAddr
        0xE2803C02u,  // add  r3, r0, #0x200
        0xE1D310B2u,  // ldrh r1, [r3, #2]      ; REG_IF
        0xE1C310B2u,  // strh r1, [r3, #2]      ; ack IF
        0xE3822C7Fu,  // orr  r2, r2, #0x7F00
        0xE1D21FB8u,  // ldrh r1, [r2, #0xF8]   ; 0x03007FF8 INTR_CHECK
        0xE3811001u,  // orr  r1, r1, #1        ; VBlank bit
        0xE1C21FB8u,  // strh r1, [r2, #0xF8]
        0xE3A02403u,  // mov  r2, #0x03000000
        0xE5921204u,  // ldr  r1, [r2, #0x204]  ; kIrqCountAddr
        0xE2811001u,  // add  r1, r1, #1
        0xE5821204u,  // str  r1, [r2, #0x204]
        0xE12FFF1Eu,  // bx   lr
    };
    for (uint32_t i = 0; i < sizeof(kCode) / sizeof(kCode[0]); ++i) {
        bus.write32(kIrqHandlerAddr + i * 4u, kCode[i]);
    }
    bus.write32(0x03007FFCu, kIrqHandlerAddr);
    // Thumb `b .` landing pad so any stray branch stays inside IWRAM.
    bus.write16(kSwiReturnPc, 0xE7FEu);
}

void init_machine(gba::GbaBus& bus, gba::GbaPpu& ppu) {
    bus.io().set_bus(&bus);
    bus.io().set_ppu(&ppu);
    gbarecomp::set_active_bus(&bus);
    gbarecomp::set_active_ppu(&ppu);
    runtime_init(&bus);
    gba::bios_hle_set_mode(gba::BiosHleMode::On);
    gba::bios_hle_set_full_fallback(true);
    gba::bios_hle_boot_skip(0x08000000u);
    install_iwram_vblank_irq_handler(bus);

    // Enable VBlank IRQ in DISPSTAT, IE, and IME.
    bus_write_u16(0x04000004u, 0x0008u);
    bus_write_u16(0x04000200u, 0x0001u);
    bus_write_u16(0x04000208u, 0x0001u);
}

void teardown_machine() {
    gba::bios_hle_set_full_fallback(false);
    gba::bios_hle_set_mode(gba::BiosHleMode::Off);
    gbarecomp::set_active_bus(nullptr);
    gbarecomp::set_active_ppu(nullptr);
    runtime_shutdown();
}

// A multi-thousand-cycle HLE SWI (e.g. CpuFastSet) crossing VBlank with IRQs enabled
// must deliver VBlank on the VBlank scanline edge rather than starving it until the
// entire SWI cycle cost finishes.
bool test_long_swi_delivers_vblank_on_time() {
    auto bus = std::make_unique<gba::GbaBus>();
    auto ppu = std::make_unique<gba::GbaPpu>();
    init_machine(*bus, *ppu);

    // Source & destination buffers in EWRAM; 8192 words = 16394 cycles (~13.3 scanlines).
    constexpr uint32_t kSrc = 0x02000000u;
    constexpr uint32_t kDst = 0x02010000u;
    bus->write32(kSrc, 0x12345678u);

    // Advance just short of VBlank (scanline 159, ~200 cycles before scanline 160).
    constexpr uint32_t kCyclesToJustBeforeVBlank = 160u * 1232u - 200u;
    runtime_tick(kCyclesToJustBeforeVBlank);
    runtime_mmio_catch_up();
    const uint16_t vc_before = ppu->vcount();
    const uint32_t irqs_before = bus->read32(kIrqCountAddr);

    // Invoke Thumb SWI #0x0C (CpuFastSet fill of 8192 words, CPSR.I == 0).
    g_cpu.cpsr = 0x0000003Fu;  // System mode, Thumb, IRQ enabled
    g_cpu.R[0] = kSrc;
    g_cpu.R[1] = kDst;
    g_cpu.R[2] = (1u << 24) | 8192u;
    g_cpu.R[15] = kSwiReturnPc;
    runtime_swi(0x0Cu);

    const uint32_t irqs_after = bus->read32(kIrqCountAddr);
    const uint32_t vcount_at_irq = bus->read32(kVCountOutAddr);
    const uint32_t copied_word = bus->read32(kDst + 0x7FFCu);
    const bool ok = vc_before == 159u && irqs_before == 0u && irqs_after == 1u &&
                    vcount_at_irq == 160u && copied_word == 0x12345678u;
    std::printf("long_swi_vblank: vc_before=%u irqs=%u vcount_at_irq=%u copied=0x%08X -> %s\n",
                vc_before, irqs_after, vcount_at_irq, copied_word, ok ? "PASS" : "FAIL");

    teardown_machine();
    return ok;
}

// VBlankIntrWait / HALT must step to exact hardware event horizons (zero 64-cycle
// polling quantization jitter across frames).
bool test_vblank_intr_wait_zero_jitter() {
    auto bus = std::make_unique<gba::GbaBus>();
    auto ppu = std::make_unique<gba::GbaPpu>();
    init_machine(*bus, *ppu);

    // Misalign by an odd cycle count that is not a multiple of 64.
    runtime_tick(37u);

    g_cpu.cpsr = 0x0000003Fu;
    g_cpu.R[15] = kSwiReturnPc;
    runtime_swi(0x05u);  // VBlankIntrWait (frame 0)
    const uint64_t cyc0 = g_runtime_cycles;
    const uint32_t vc0 = bus->read32(kVCountOutAddr);

    runtime_tick(19u);   // Arbitrary non-multiple-of-64 active work before next wait
    runtime_swi(0x05u);  // VBlankIntrWait (frame 1)
    const uint64_t cyc1 = g_runtime_cycles;
    const uint32_t vc1 = bus->read32(kVCountOutAddr);

    runtime_tick(513u);
    runtime_swi(0x05u);  // VBlankIntrWait (frame 2)
    const uint64_t cyc2 = g_runtime_cycles;
    const uint32_t vc2 = bus->read32(kVCountOutAddr);

    const uint64_t d1 = cyc1 - cyc0;
    const uint64_t d2 = cyc2 - cyc1;
    const bool ok = (bus->read32(kIrqCountAddr) == 3u) &&
                    vc0 == 160u && vc1 == 160u && vc2 == 160u &&
                    d1 == 280896ull && d2 == 280896ull;
    std::printf("vblank_intr_wait: d1=%llu d2=%llu vc=(%u,%u,%u) -> %s\n",
                static_cast<unsigned long long>(d1),
                static_cast<unsigned long long>(d2),
                vc0, vc1, vc2, ok ? "PASS" : "FAIL");

    teardown_machine();
    return ok;
}

}  // namespace

int main() {
    bool ok = true;
    ok = test_long_swi_delivers_vblank_on_time() && ok;
    ok = test_vblank_intr_wait_zero_jitter() && ok;
    return ok ? 0 : 1;
}
