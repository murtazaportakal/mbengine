//! Input state tracking.

#[derive(Debug, Clone)]
pub struct Input {
    pub keys: [bool; 256],
    pub keys_prev: [bool; 256],
    pub mouse_x: i32,
    pub mouse_y: i32,
}

impl Default for Input {
    fn default() -> Self {
        Self::new()
    }
}

impl Input {
    pub fn new() -> Self {
        Self {
            keys: [false; 256],
            keys_prev: [false; 256],
            mouse_x: 0,
            mouse_y: 0,
        }
    }

    /// Called at the beginning of the frame to capture the previous frame's state.
    pub fn update(&mut self) {
        self.keys_prev.copy_from_slice(&self.keys);
    }

    /// Is the key currently held down?
    pub fn is_key_down(&self, vk_code: usize) -> bool {
        if vk_code < 256 {
            self.keys[vk_code]
        } else {
            false
        }
    }

    /// Was the key pressed exactly this frame?
    pub fn is_key_pressed(&self, vk_code: usize) -> bool {
        if vk_code < 256 {
            self.keys[vk_code] && !self.keys_prev[vk_code]
        } else {
            false
        }
    }

    /// Was the key released exactly this frame?
    pub fn is_key_released(&self, vk_code: usize) -> bool {
        if vk_code < 256 {
            !self.keys[vk_code] && self.keys_prev[vk_code]
        } else {
            false
        }
    }
}
