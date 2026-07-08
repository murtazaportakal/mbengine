fn main() {
    #[repr(C)]
    pub struct PushConstants {
        pub world: [[f32; 4]; 4],
        pub metallic: f32,
        pub roughness: f32,
        pub padding: [f32; 2],
        pub color: [f32; 4],
    }
    
    println!("Size: {}", std::mem::size_of::<PushConstants>());
    
    let dummy = PushConstants {
        world: [[0.0; 4]; 4],
        metallic: 0.0,
        roughness: 0.0,
        padding: [0.0; 2],
        color: [0.0; 4],
    };
    
    let base = &dummy as *const _ as usize;
    let world_offset = &dummy.world as *const _ as usize - base;
    let metallic_offset = &dummy.metallic as *const _ as usize - base;
    let roughness_offset = &dummy.roughness as *const _ as usize - base;
    let padding_offset = &dummy.padding as *const _ as usize - base;
    let color_offset = &dummy.color as *const _ as usize - base;
    
    println!("world: {}", world_offset);
    println!("metallic: {}", metallic_offset);
    println!("roughness: {}", roughness_offset);
    println!("padding: {}", padding_offset);
    println!("color: {}", color_offset);
}
