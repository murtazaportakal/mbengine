use crate::math::vec::Vec2;

#[derive(Clone, Copy, Debug)]
pub struct UiColor {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl UiColor {
    pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self { Self { r, g, b, a } }
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self { Self { r, g, b, a: 255 } }
    pub const WHITE: Self = Self::rgb(255, 255, 255);
    pub const BLACK: Self = Self::rgb(0, 0, 0);
}

#[derive(Clone, Copy, Debug)]
pub struct UiStyle {
    pub bg_color: UiColor,
    pub border_color: UiColor,
    pub border_width: f32,
    pub text_color: UiColor,
    pub rounding: f32,
}

pub const SLATE_BASE: UiStyle = UiStyle {
    bg_color: UiColor::rgb(56, 56, 56),
    border_color: UiColor::rgb(35, 35, 35),
    border_width: 1.0,
    text_color: UiColor::rgb(224, 224, 224),
    rounding: 2.0,
};

pub const SLATE_SECONDARY: UiStyle = UiStyle {
    bg_color: UiColor::rgb(40, 40, 40),
    border_color: UiColor::rgb(30, 30, 30),
    border_width: 1.0,
    text_color: UiColor::rgb(210, 210, 210),
    rounding: 2.0,
};

pub const SLATE_INPUT_BG: UiStyle = UiStyle {
    bg_color: UiColor::rgb(42, 42, 42),
    border_color: UiColor::rgb(25, 25, 25),
    border_width: 1.0,
    text_color: UiColor::rgb(224, 224, 224),
    rounding: 2.0,
};

pub const SLATE_ACCENT: UiStyle = UiStyle {
    bg_color: UiColor::rgb(60, 110, 180),
    border_color: UiColor::rgb(80, 130, 200),
    border_width: 1.0,
    text_color: UiColor::rgb(255, 255, 255),
    rounding: 2.0,
};

#[derive(Clone, Copy, Debug)]
pub struct UiRect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl UiRect {
    pub fn contains(&self, p: Vec2) -> bool {
        p.x >= self.x && p.x <= self.x + self.w && p.y >= self.y && p.y <= self.y + self.h
    }
}

pub enum DrawCommand {
    Quad {
        rect: UiRect,
        color: UiColor,
        rounding: f32,
    },
    Line {
        p1: Vec2,
        p2: Vec2,
        color: UiColor,
        thickness: f32,
    },
    Image {
        rect: UiRect,
        uv_min: Vec2,
        uv_max: Vec2,
        color: UiColor,
        texture_id: u32,
    },
    Text {
        pos: Vec2,
        text: crate::containers::FixedString<128>,
        color: UiColor,
        font_size: f32,
    },
}

pub struct UiContext {
    pub draw_commands: Vec<DrawCommand>,
    
    // Input state
    pub mouse_pos: Vec2,
    pub last_mouse_pos: Vec2,
    pub mouse_delta: Vec2,
    pub mouse_down: bool,
    pub mouse_pressed: bool,
    pub mouse_released: bool,
    
    pub hot_item: u64,
    pub active_item: u64,
    
    // Auto-layout
    pub current_rect: UiRect,
    pub cursor: Vec2,
    pub is_horizontal: bool,
    pub row_height: f32,
    pub layout_stack: Vec<LayoutState>,
}

#[derive(Clone, Copy, Debug)]
pub struct LayoutState {
    pub current_rect: UiRect,
    pub cursor: Vec2,
    pub is_horizontal: bool,
    pub row_height: f32,
}

impl Default for UiContext {
    fn default() -> Self {
        Self::new()
    }
}

impl UiContext {
    pub fn new() -> Self {
        Self {
            draw_commands: Vec::with_capacity(4096),
            mouse_pos: Vec2::new(0.0, 0.0),
            last_mouse_pos: Vec2::new(0.0, 0.0),
            mouse_delta: Vec2::new(0.0, 0.0),
            mouse_down: false,
            mouse_pressed: false,
            mouse_released: false,
            hot_item: 0,
            active_item: 0,
            current_rect: UiRect { x: 0.0, y: 0.0, w: 0.0, h: 0.0 },
            cursor: Vec2::new(0.0, 0.0),
            is_horizontal: false,
            row_height: 0.0,
            layout_stack: Vec::with_capacity(32),
        }
    }

    pub fn begin_frame(&mut self, mouse_pos: Vec2, mouse_down: bool) {
        self.draw_commands.clear();
        
        let was_down = self.mouse_down;
        self.last_mouse_pos = self.mouse_pos;
        self.mouse_pos = mouse_pos;
        self.mouse_delta = Vec2::new(
            self.mouse_pos.x - self.last_mouse_pos.x,
            self.mouse_pos.y - self.last_mouse_pos.y,
        );
        self.mouse_down = mouse_down;
        self.mouse_pressed = !was_down && mouse_down;
        self.mouse_released = was_down && !mouse_down;
        
        self.hot_item = 0;
    }

    pub fn end_frame(&mut self) {
        if self.mouse_released {
            self.active_item = 0;
        }
    }

    pub fn add_line(&mut self, p1: Vec2, p2: Vec2, color: UiColor, thickness: f32) {
        self.draw_commands.push(DrawCommand::Line { p1, p2, color, thickness });
    }
    
    pub fn hash_id(&self, label: &str) -> u64 {
        let mut hash: u64 = 0xcbf29ce484222325;
        for b in label.bytes() {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }
    
    pub fn begin_panel(&mut self, _id: u64, rect: UiRect, style: &UiStyle) {
        // Draw Border
        if style.border_width > 0.0 {
            self.draw_commands.push(DrawCommand::Quad {
                rect,
                color: style.border_color,
                rounding: style.rounding,
            });
        }
        // Draw Background
        let mut bg_rect = rect;
        bg_rect.x += style.border_width;
        bg_rect.y += style.border_width;
        bg_rect.w -= style.border_width * 2.0;
        bg_rect.h -= style.border_width * 2.0;
        
        self.draw_commands.push(DrawCommand::Quad {
            rect: bg_rect,
            color: style.bg_color,
            rounding: style.rounding,
        });

        self.current_rect = bg_rect;
    }

    pub fn end_panel(&mut self) {
        // Just scopes logically for now
    }

    pub fn push_layout(&mut self) {
        self.layout_stack.push(LayoutState {
            current_rect: self.current_rect,
            cursor: self.cursor,
            is_horizontal: self.is_horizontal,
            row_height: self.row_height,
        });
    }

    pub fn pop_layout(&mut self) {
        if let Some(state) = self.layout_stack.pop() {
            self.current_rect = state.current_rect;
            self.cursor = state.cursor;
            self.is_horizontal = state.is_horizontal;
            self.row_height = state.row_height;
        }
    }

    pub fn begin_vertical_layout(&mut self, start_rect: UiRect) {
        self.push_layout();
        self.current_rect = start_rect;
        self.cursor = Vec2::new(start_rect.x + 12.0, start_rect.y + 12.0);
        self.is_horizontal = false;
        self.row_height = 0.0;
    }

    pub fn end_vertical_layout(&mut self) {
        self.pop_layout();
    }

    pub fn begin_horizontal_layout(&mut self, start_rect: UiRect) {
        self.push_layout();
        self.current_rect = start_rect;
        self.cursor = Vec2::new(start_rect.x + 12.0, start_rect.y + 8.0);
        self.is_horizontal = true;
        self.row_height = 0.0;
    }

    pub fn end_horizontal_layout(&mut self) {
        self.pop_layout();
    }

    pub fn indent_cursor(&mut self, amount: f32) {
        self.cursor.x += amount;
    }

    pub fn unindent_cursor(&mut self, amount: f32) {
        self.cursor.x -= amount;
    }

    fn advance_cursor(&mut self, width: f32, height: f32) {
        if self.is_horizontal {
            self.cursor.x += width + 4.0;
            if height > self.row_height {
                self.row_height = height;
            }
        } else {
            self.cursor.y += height + 4.0;
        }
    }

    pub fn image(&mut self, rect: UiRect, texture_id: u32) {
        self.draw_commands.push(DrawCommand::Image {
            rect,
            uv_min: Vec2::new(0.0, 0.0),
            uv_max: Vec2::new(1.0, 1.0),
            color: UiColor::WHITE,
            texture_id,
        });
    }

    pub fn label(&mut self, text: &str) {
        let width = text.len() as f32 * 7.0; // rough approximation for smaller font
        let height = 20.0;
        
        self.draw_commands.push(DrawCommand::Text {
            pos: Vec2::new(self.cursor.x, self.cursor.y + 14.0),
            text: crate::containers::FixedString::<128>::try_from_str(text).unwrap_or_default(),
            color: SLATE_BASE.text_color,
            font_size: 14.0,
        });

        self.advance_cursor(width, height);
    }

    pub fn label_color(&mut self, text: &str, color: UiColor) {
        let width = text.len() as f32 * 7.0; // rough approximation for smaller font
        let height = 20.0;
        
        self.draw_commands.push(DrawCommand::Text {
            pos: Vec2::new(self.cursor.x, self.cursor.y + 14.0),
            text: crate::containers::FixedString::<128>::try_from_str(text).unwrap_or_default(),
            color,
            font_size: 14.0,
        });

        self.advance_cursor(width, height);
    }

    pub fn selectable_label(&mut self, text: &str, is_selected: bool) -> bool {
        let width = self.current_rect.w - 24.0; // fill remaining width with padding
        let height = 20.0;
        let rect = UiRect {
            x: self.cursor.x,
            y: self.cursor.y,
            w: width,
            h: height,
        };

        let hovered = rect.contains(self.mouse_pos);
        let clicked = hovered && self.mouse_pressed;

        let bg_color = if is_selected {
            SLATE_ACCENT.bg_color
        } else if hovered {
            UiColor::rgb(60, 60, 60)
        } else {
            UiColor::rgba(0, 0, 0, 0) // Transparent
        };

        if bg_color.a > 0 {
            self.draw_commands.push(DrawCommand::Quad {
                rect,
                color: bg_color,
                rounding: 2.0,
            });
        }

        self.draw_commands.push(DrawCommand::Text {
            pos: Vec2::new(rect.x + 8.0, rect.y + 14.0),
            text: crate::containers::FixedString::<128>::try_from_str(text).unwrap_or_default(),
            color: if is_selected { SLATE_ACCENT.text_color } else { SLATE_BASE.text_color },
            font_size: 14.0,
        });

        self.advance_cursor(width, height);
        clicked
    }

    pub fn layout_button(&mut self, id: u64, text: &str, style: &UiStyle) -> bool {
        let width = text.len() as f32 * 7.0 + 16.0;
        let height = 20.0;
        let rect = UiRect {
            x: self.cursor.x,
            y: self.cursor.y,
            w: width,
            h: height,
        };

        let hovered = rect.contains(self.mouse_pos);
        if hovered {
            self.hot_item = id;
            if self.active_item == 0 && self.mouse_pressed {
                self.active_item = id;
            }
        }

        let mut bg_color = style.bg_color;
        if self.hot_item == id {
            if self.active_item == id {
                bg_color = SLATE_ACCENT.bg_color;
            } else {
                bg_color = UiColor::rgb(bg_color.r.saturating_add(20), bg_color.g.saturating_add(20), bg_color.b.saturating_add(20));
            }
        }

        // Draw border
        if style.border_width > 0.0 {
            self.draw_commands.push(DrawCommand::Quad {
                rect,
                color: style.border_color,
                rounding: style.rounding,
            });
        }
        
        // Draw bg
        let mut inner_rect = rect;
        inner_rect.x += style.border_width;
        inner_rect.y += style.border_width;
        inner_rect.w -= style.border_width * 2.0;
        inner_rect.h -= style.border_width * 2.0;
        
        self.draw_commands.push(DrawCommand::Quad {
            rect: inner_rect,
            color: bg_color,
            rounding: style.rounding,
        });

        self.draw_commands.push(DrawCommand::Text {
            pos: Vec2::new(rect.x + 8.0, rect.y + 14.0),
            text: crate::containers::FixedString::<128>::try_from_str(text).unwrap_or_default(),
            color: style.text_color,
            font_size: 14.0,
        });

        self.advance_cursor(width, height);

        self.mouse_released && self.hot_item == id && self.active_item == id
    }

    pub fn collapsing_header(&mut self, label: &str, is_expanded: &mut bool) -> bool {
        let id = self.hash_id(label);
        let width = self.current_rect.w - 8.0;
        let height = 22.0;
        let rect = UiRect {
            x: self.current_rect.x + 4.0, // Slight inner padding
            y: self.cursor.y,
            w: width,
            h: height,
        };

        let hovered = rect.contains(self.mouse_pos);
        if hovered {
            self.hot_item = id;
            if self.active_item == 0 && self.mouse_pressed {
                self.active_item = id;
            }
        }

        if self.mouse_released && self.hot_item == id && self.active_item == id {
            *is_expanded = !*is_expanded;
        }

        let bg_color = if hovered { UiColor::rgb(65, 65, 65) } else { UiColor::rgb(55, 55, 55) };

        self.draw_commands.push(DrawCommand::Quad {
            rect,
            color: bg_color,
            rounding: 2.0,
        });

        let icon = if *is_expanded { "▼" } else { "▶" };
        let mut text = crate::containers::FixedString::<128>::try_from_str(icon).unwrap_or_default();
        text.push_str(" ");
        text.push_str(label);

        self.draw_commands.push(DrawCommand::Text {
            pos: Vec2::new(rect.x + 4.0, rect.y + 15.0),
            text,
            color: UiColor::WHITE,
            font_size: 14.0,
        });

        self.cursor.y += height + 2.0;
        *is_expanded
    }

    pub fn drag_float(&mut self, label: &str, value: &mut f32) -> bool {
        use core::fmt::Write;
        let id = self.hash_id(label);
        
        let total_width = self.current_rect.w - 24.0;
        let label_width = total_width * 0.4; // 40% label
        let input_width = total_width * 0.6; // 60% input
        let height = 20.0;

        // Draw left side label
        self.draw_commands.push(DrawCommand::Text {
            pos: Vec2::new(self.cursor.x, self.cursor.y + 14.0),
            text: crate::containers::FixedString::<128>::try_from_str(label).unwrap_or_default(),
            color: SLATE_BASE.text_color,
            font_size: 14.0,
        });

        // Right side input rect
        let rect = UiRect {
            x: self.cursor.x + label_width,
            y: self.cursor.y,
            w: input_width,
            h: height,
        };

        let hovered = rect.contains(self.mouse_pos);
        if hovered {
            self.hot_item = id;
            if self.active_item == 0 && self.mouse_pressed {
                self.active_item = id;
            }
        }

        let mut changed = false;
        if self.active_item == id
            && self.mouse_delta.x != 0.0 {
                *value += self.mouse_delta.x * 0.05;
                changed = true;
            }

        let mut bg_color = SLATE_INPUT_BG.bg_color;
        if self.hot_item == id || self.active_item == id {
            bg_color = UiColor::rgb(60, 60, 60);
        }

        self.draw_commands.push(DrawCommand::Quad {
            rect,
            color: bg_color,
            rounding: 2.0,
        });

        let mut display_text = crate::containers::FixedString::<128>::new();
        let _ = write!(display_text, "{:.3}", *value);

        self.draw_commands.push(DrawCommand::Text {
            pos: Vec2::new(rect.x + 8.0, rect.y + 14.0),
            text: display_text,
            color: SLATE_INPUT_BG.text_color,
            font_size: 14.0,
        });

        self.advance_cursor(total_width, height);
        changed
    }

    pub fn checkbox(&mut self, label: &str, value: &mut bool) -> bool {
        let id = self.hash_id(label);
        
        let total_width = self.current_rect.w - 24.0;
        let label_width = total_width * 0.4;
        let input_width = total_width * 0.6;
        let height = 20.0;
        
        // Draw left side label
        self.draw_commands.push(DrawCommand::Text {
            pos: Vec2::new(self.cursor.x, self.cursor.y + 14.0),
            text: crate::containers::FixedString::<128>::try_from_str(label).unwrap_or_default(),
            color: SLATE_BASE.text_color,
            font_size: 14.0,
        });

        let box_size = 14.0;
        let box_rect = UiRect {
            x: self.cursor.x + label_width,
            y: self.cursor.y + (height - box_size) / 2.0,
            w: box_size,
            h: box_size,
        };

        // We capture clicks anywhere on the right side row
        let click_rect = UiRect {
            x: self.cursor.x + label_width,
            y: self.cursor.y,
            w: input_width,
            h: height,
        };

        let hovered = click_rect.contains(self.mouse_pos);
        let mut clicked = false;
        if hovered {
            self.hot_item = id;
            if self.active_item == 0 && self.mouse_pressed {
                self.active_item = id;
            }
        }

        if self.mouse_released && self.hot_item == id && self.active_item == id {
            *value = !*value;
            clicked = true;
        }

        let mut bg_color = SLATE_INPUT_BG.bg_color;
        if self.hot_item == id || self.active_item == id {
            bg_color = UiColor::rgb(60, 60, 60);
        }

        // Draw checkbox bg
        self.draw_commands.push(DrawCommand::Quad {
            rect: box_rect,
            color: bg_color,
            rounding: 2.0,
        });

        // Draw checkmark
        if *value {
            let check_rect = UiRect {
                x: box_rect.x + 3.0,
                y: box_rect.y + 3.0,
                w: 8.0,
                h: 8.0,
            };
            self.draw_commands.push(DrawCommand::Quad {
                rect: check_rect,
                color: SLATE_ACCENT.bg_color,
                rounding: 1.0,
            });
        }

        self.advance_cursor(total_width, height);
        clicked
    }
}

// ── Builders ────────────────────────────────────────────────────────────────
// Taking inspiration from Fyrox's fluent builder pattern.

pub struct PanelBuilder<'a> {
    ctx: &'a mut UiContext,
    id: u64,
    rect: UiRect,
    style: &'a UiStyle,
}

impl<'a> PanelBuilder<'a> {
    pub fn new(ctx: &'a mut UiContext, id: u64) -> Self {
        Self {
            ctx,
            id,
            rect: UiRect { x: 0.0, y: 0.0, w: 100.0, h: 100.0 },
            style: &SLATE_BASE,
        }
    }
    
    pub fn rect(mut self, rect: UiRect) -> Self { self.rect = rect; self }
    pub fn style(mut self, style: &'a UiStyle) -> Self { self.style = style; self }
    
    pub fn begin(self) {
        self.ctx.begin_panel(self.id, self.rect, self.style);
    }
}

pub struct ButtonBuilder<'a> {
    ctx: &'a mut UiContext,
    id: u64,
    text: &'a str,
    style: &'a UiStyle,
}

impl<'a> ButtonBuilder<'a> {
    pub fn new(ctx: &'a mut UiContext, id: u64) -> Self {
        Self {
            ctx,
            id,
            text: "",
            style: &SLATE_BASE,
        }
    }
    
    pub fn text(mut self, text: &'a str) -> Self { self.text = text; self }
    pub fn style(mut self, style: &'a UiStyle) -> Self { self.style = style; self }
    
    pub fn build(self) -> bool {
        self.ctx.layout_button(self.id, self.text, self.style)
    }
}
