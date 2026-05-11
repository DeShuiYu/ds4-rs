//! Build script for the DS4 Rust inference engine.
//!
//! On macOS: if the `metal` feature is enabled, compiles the Metal C/ObjC
//! runtime (`ds4_metal.m`) as a static library and copies the Metal shader
//! (`.metal`) source files to the output directory for runtime loading.
//! This requires macOS 14.0+ (Sonoma) and Xcode 15+.
//!
//! Without the `metal` feature (the default), defines `ds4_no_metal` to
//! exclude Metal-dependent code and builds a CPU-only binary.

use std::env;
use std::path::Path;

fn main() {
    // Respect the Cargo feature flag.
    let metal_enabled = env::var("CARGO_FEATURE_METAL").is_ok();

    #[cfg(target_os = "macos")]
    {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let out_dir_str = env::var("OUT_DIR").unwrap();
        let out_dir = Path::new(&out_dir_str);

        if metal_enabled {
            // ── Metal shader sources ──────────────────────────────────────
            let metal_src = manifest_dir.join("..").join("metal");
            let metal_dst = out_dir.join("metal");

            if metal_src.exists() {
                std::fs::create_dir_all(&metal_dst).ok();
                for entry in std::fs::read_dir(&metal_src).unwrap() {
                    let entry = entry.unwrap();
                    let path = entry.path();
                    if path.extension().map_or(false, |e| e == "metal") {
                        let dest = metal_dst.join(path.file_name().unwrap());
                        std::fs::copy(&path, &dest).ok();
                        println!("cargo:rerun-if-changed={}", path.display());
                    }
                }
            }

            // ── Compile ds4_metal.m ───────────────────────────────────────
            let metal_m = manifest_dir.join("..").join("ds4_metal.m");
            if metal_m.exists() {
                // Metal residency sets require macOS 14.0+
                env::set_var("MACOSX_DEPLOYMENT_TARGET", "14.0");
                cc::Build::new()
                    .file(&metal_m)
                    .flag("-fobjc-arc")
                    .flag("-O3")
                    .flag("-Wno-nullability-completeness")
                    .flag("-Wno-unguarded-availability-new")
                    .flag("-framework")
                    .flag("Foundation")
                    .flag("-framework")
                    .flag("Metal")
                    .flag("-framework")
                    .flag("MetalKit")
                    .flag("-framework")
                    .flag("CoreGraphics")
                    .compile("ds4_metal");

                println!("cargo:rerun-if-changed={}", metal_m.display());
                println!("cargo:rerun-if-changed=../ds4_metal.h");
                return;
            }
        }

        // Fallback: no Metal available (default).
        println!("cargo:warning=ds4: Metal backend requires macOS 14+ and the `metal` feature.");
        println!("cargo:warning=ds4: Building CPU-only reference backend.");
        println!("cargo:rustc-cfg=ds4_no_metal");
    }

    #[cfg(not(target_os = "macos"))]
    {
        println!("cargo:rustc-cfg=ds4_no_metal");
    }
}
