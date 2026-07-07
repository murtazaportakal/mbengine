
use crate::ui::context::UiRect;

pub struct UiLayout {
    pub start_x: f32,
    pub cursor_x: f32,
    pub cursor_y: f32,
    pub width: f32,
    pub row_height: f32,
}

impl UiLayout {
    pub fn new(x: f32, y: f32, width: f32) -> Self {
        Self {
            start_x: x,
            cursor_x: x,
            cursor_y: y,
            width,
            row_height: 0.0,
        }
    }
    
    pub fn next_rect(&mut self, item_width: f32, item_height: f32) -> UiRect {
        if self.cursor_x + item_width > self.start_x + self.width && self.cursor_x > self.start_x {
            self.new_line();
        }
        
        let rect = UiRect {
            x: self.cursor_x,
            y: self.cursor_y,
            w: item_width,
            h: item_height,
        };
        
        self.cursor_x += item_width + 4.0;
        if item_height > self.row_height {
            self.row_height = item_height;
        }
        
        rect
    }
    
    pub fn new_line(&mut self) {
        self.cursor_x = self.start_x;
        self.cursor_y += self.row_height + 4.0;
        self.row_height = 0.0;
    }
}
