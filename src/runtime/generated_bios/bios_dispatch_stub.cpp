#include <cstdint>
#include "runtime_arm.h"

// Clean-checkout fallback used when no recompiled BIOS is present. Includes
// native implementations of the standard ARM7TDMI GBA IRQ trampoline
// (0x18 -> 0x128..0x13C) so HLE-BIOS execution never hits interpreter bridges
// during VBlank / hardware interrupts.
struct DispatchEntry {
    std::uint32_t addr;
    std::uint8_t thumb;
    std::uint8_t resume;
    void (*fn)(void);
};

static void hle_bios_00000018() {
    const std::uint32_t sp = g_cpu.R[13] - 24u;
    g_cpu.R[13] = sp;
    bus_write_u32(sp + 0u,  g_cpu.R[0]);
    bus_write_u32(sp + 4u,  g_cpu.R[1]);
    bus_write_u32(sp + 8u,  g_cpu.R[2]);
    bus_write_u32(sp + 12u, g_cpu.R[3]);
    bus_write_u32(sp + 16u, g_cpu.R[12]);
    bus_write_u32(sp + 20u, g_cpu.R[14]);
    g_cpu.R[0] = 0x04000000u;
    g_cpu.R[14] = 0x00000138u;
    const std::uint32_t handler = bus_read_u32(0x03FFFFFCu);
    g_cpu.R[15] = handler & ~1u;
    runtime_tick(12u);
    GBARECOMP_TAIL_DISPATCH(handler);
}

static void hle_bios_00000138() {
    const std::uint32_t sp = g_cpu.R[13];
    g_cpu.R[0]  = bus_read_u32(sp + 0u);
    g_cpu.R[1]  = bus_read_u32(sp + 4u);
    g_cpu.R[2]  = bus_read_u32(sp + 8u);
    g_cpu.R[3]  = bus_read_u32(sp + 12u);
    g_cpu.R[12] = bus_read_u32(sp + 16u);
    g_cpu.R[14] = bus_read_u32(sp + 20u);
    g_cpu.R[13] = sp + 24u;
    runtime_tick(10u);
    runtime_exception_return(g_cpu.R[14] - 4u);
}

extern "C" const DispatchEntry kBiosDispatchTable[] = {
    {0x00000018u, 0u, 0u, hle_bios_00000018},
    {0x00000128u, 0u, 0u, hle_bios_00000018},
    {0x00000138u, 0u, 0u, hle_bios_00000138},
};
extern "C" const unsigned kBiosDispatchTableLen = 3;
