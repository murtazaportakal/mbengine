use ab_glyph::{FontRef, Font as AbFont, PxScale, ScaleFont};

pub struct Font {
    pub texture_data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub glyphs: [GlyphInfo; 128],
    pub line_height: f32,
}

#[derive(Clone, Copy, Default)]
pub struct GlyphInfo {
    pub u_min: f32,
    pub v_min: f32,
    pub u_max: f32,
    pub v_max: f32,
    pub advance: f32,
    pub offset_x: f32,
    pub offset_y: f32,
    pub size_x: f32,
    pub size_y: f32,
}

impl Font {
    pub fn load_ascii(ttf_bytes: &[u8], px_size: f32) -> Option<Self> {
        let font = FontRef::try_from_slice(ttf_bytes).ok()?;
        let scale = PxScale::from(px_size);
        
        let width = 1024;
        let height = 1024;
        // RGBA texture
        let mut texture_data = vec![0; (width * height * 4) as usize];
        let mut glyphs = [GlyphInfo::default(); 128];
        
        let mut cur_x = 0;
        let mut cur_y = 0;
        let line_height = px_size.ceil() as u32 + 4;
        
        for c in 32..127u8 {
            let glyph_id = font.glyph_id(c as char);
            let advance = font.as_scaled(scale).h_advance(glyph_id);
            
            let q = glyph_id.with_scale_and_position(scale, ab_glyph::point(0.0, 0.0));
            if let Some(outlined) = font.outline_glyph(q) {
                let bounds = outlined.px_bounds();
                let w = bounds.width() as u32;
                let h = bounds.height() as u32;
                
                if cur_x + w + 2 > width {
                    cur_x = 0;
                    cur_y += line_height;
                }
                
                outlined.draw(|x, y, v| {
                    let px = cur_x + x;
                    let py = cur_y + y;
                    if px < width && py < height {
                        let idx = ((py * width + px) * 4) as usize;
                        let val = (v * 255.0) as u8;
                        texture_data[idx] = val;
                        texture_data[idx + 1] = val;
                        texture_data[idx + 2] = val;
                        texture_data[idx + 3] = val; // Store coverage in alpha channel
                    }
                });
                
                glyphs[c as usize] = GlyphInfo {
                    u_min: cur_x as f32 / width as f32,
                    v_min: cur_y as f32 / height as f32,
                    u_max: (cur_x + w) as f32 / width as f32,
                    v_max: (cur_y + h) as f32 / height as f32,
                    advance,
                    offset_x: bounds.min.x,
                    offset_y: bounds.min.y,
                    size_x: w as f32,
                    size_y: h as f32,
                };
                
                cur_x += w + 2;
            } else {
                glyphs[c as usize].advance = advance;
            }
        }
        
        let v_metrics = font.as_scaled(scale).ascent() - font.as_scaled(scale).descent();
        
        Some(Self {
            texture_data,
            width,
            height,
            glyphs,
            line_height: v_metrics,
        })
    }
}
