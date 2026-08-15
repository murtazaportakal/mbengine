fn main() {
    println!("MeshletData size: {}", std::mem::size_of::<MeshletData>());
}

#[repr(C)]
struct MeshletData {
    center: [f32; 3],
    radius: f32,
    cone_axis: [f32; 3],
    cone_cutoff: f32,
    index_offset: u32,
    triangle_count: u32,
    _pad: [u32; 2],
}
