// gba_bios.cpp — see gba_bios.h.

#include "gba_bios.h"

#include <cstdio>
#include <cstring>

#include "sha1.h"

namespace gba {

GbaBios::GbaBios()  = default;
GbaBios::~GbaBios() = default;

bool GbaBios::load_from_file(const std::string& path,
                             const std::string& expected_sha1,
                             std::string* err) {
    loaded_ = false;
    hle_stub_ = false;
    sha1_hex_.clear();

    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        if (err) *err = "open failed: " + path;
        return false;
    }
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (sz < 0 || static_cast<std::size_t>(sz) != kSize) {
        if (err) {
            *err = "BIOS size mismatch: expected " +
                   std::to_string(kSize) + " bytes, got " +
                   (sz < 0 ? "<seek failed>" : std::to_string(sz));
        }
        std::fclose(f);
        return false;
    }
    std::size_t got = std::fread(bytes_.data(), 1, kSize, f);
    std::fclose(f);
    if (got != kSize) {
        if (err) *err = "short read from BIOS file: " + path;
        return false;
    }

    auto digest = sha1(bytes_.data(), kSize);
    sha1_hex_ = digest.hex();
    if (!expected_sha1.empty() && sha1_hex_ != expected_sha1) {
        if (err) {
            *err = "BIOS SHA-1 mismatch: got " + sha1_hex_ +
                   " expected " + expected_sha1;
        }
        return false;
    }

    loaded_ = true;
    return true;
}

void GbaBios::load_hle_stub() {
    bytes_.fill(0);
    auto put32 = [&](uint32_t off, uint32_t word) {
        bytes_[off + 0] = static_cast<uint8_t>(word & 0xFFu);
        bytes_[off + 1] = static_cast<uint8_t>((word >> 8) & 0xFFu);
        bytes_[off + 2] = static_cast<uint8_t>((word >> 16) & 0xFFu);
        bytes_[off + 3] = static_cast<uint8_t>((word >> 24) & 0xFFu);
    };
    // Standard ARM7TDMI GBA IRQ vector (0x18 -> 0x128) and trampoline:
    //   0x018: b     0x128
    //   0x128: stmfd sp!, {r0-r3, r12, lr}
    //   0x12C: mov   r0, #0x04000000
    //   0x130: add   lr, pc, #0
    //   0x134: ldr   pc, [r0, #-4]
    //   0x138: ldmfd sp!, {r0-r3, r12, lr}
    //   0x13C: subs  pc, lr, #4
    put32(0x018, 0xEA000042u);
    put32(0x128, 0xE92D500Fu);
    put32(0x12C, 0xE3A00301u);
    put32(0x130, 0xE28FE000u);
    put32(0x134, 0xE510F004u);
    put32(0x138, 0xE8BD500Fu);
    put32(0x13C, 0xE25EF004u);
    sha1_hex_ = "hle-stub";
    hle_stub_ = true;
    loaded_ = true;
}

uint8_t GbaBios::read8(uint32_t off) const {
    if (!loaded_ || off >= kSize) return 0;
    return bytes_[off];
}

uint16_t GbaBios::read16(uint32_t off) const {
    if (!loaded_ || off + 1 >= kSize) return 0;
    const uint8_t* p = &bytes_[off];
    return static_cast<uint16_t>(p[0] | (p[1] << 8));
}

uint32_t GbaBios::read32(uint32_t off) const {
    if (!loaded_ || off + 3 >= kSize) return 0;
    const uint8_t* p = &bytes_[off];
    return static_cast<uint32_t>(p[0] |
                                 (p[1] << 8) |
                                 (p[2] << 16) |
                                 (p[3] << 24));
}

}  // namespace gba
