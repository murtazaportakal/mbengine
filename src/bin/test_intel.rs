use intel_tex_2::bc7;
use intel_tex_2::RgbaSurface;

fn main() {
    let rgba_data = vec![0u8; 16 * 16 * 4];
    let surface = RgbaSurface {
        data: &rgba_data,
        width: 16,
        height: 16,
        stride: 16 * 4,
    };
    let settings = bc7::alpha_ultra_fast_settings();
    let compressed = bc7::compress_blocks(&settings, &surface);
    println!("{}", compressed.len());
}
