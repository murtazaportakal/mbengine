use crate::platform::win32;
use egui::{Event, Key, PointerButton, Pos2, RawInput, Vec2};

pub fn translate_win32_to_egui(msg: &win32::MSG, raw_input: &mut RawInput, pixels_per_point: f32) {
    match msg.message {
        win32::WM_MOUSEMOVE => {
            let x = (msg.lParam & 0xFFFF) as i16 as f32 / pixels_per_point;
            let y = ((msg.lParam >> 16) & 0xFFFF) as i16 as f32 / pixels_per_point;
            raw_input.events.push(Event::PointerMoved(Pos2::new(x, y)));
        }
        win32::WM_LBUTTONDOWN | win32::WM_LBUTTONUP => {
            let x = (msg.lParam & 0xFFFF) as i16 as f32 / pixels_per_point;
            let y = ((msg.lParam >> 16) & 0xFFFF) as i16 as f32 / pixels_per_point;
            let pressed = msg.message == win32::WM_LBUTTONDOWN;
            raw_input.events.push(Event::PointerButton {
                pos: Pos2::new(x, y),
                button: PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::default(),
            });
        }
        win32::WM_RBUTTONDOWN | win32::WM_RBUTTONUP => {
            let x = (msg.lParam & 0xFFFF) as i16 as f32 / pixels_per_point;
            let y = ((msg.lParam >> 16) & 0xFFFF) as i16 as f32 / pixels_per_point;
            let pressed = msg.message == win32::WM_RBUTTONDOWN;
            raw_input.events.push(Event::PointerButton {
                pos: Pos2::new(x, y),
                button: PointerButton::Secondary,
                pressed,
                modifiers: egui::Modifiers::default(),
            });
        }
        win32::WM_MBUTTONDOWN | win32::WM_MBUTTONUP => {
            let x = (msg.lParam & 0xFFFF) as i16 as f32 / pixels_per_point;
            let y = ((msg.lParam >> 16) & 0xFFFF) as i16 as f32 / pixels_per_point;
            let pressed = msg.message == win32::WM_MBUTTONDOWN;
            raw_input.events.push(Event::PointerButton {
                pos: Pos2::new(x, y),
                button: PointerButton::Middle,
                pressed,
                modifiers: egui::Modifiers::default(),
            });
        }
        win32::WM_MOUSEWHEEL => {
            let delta = ((msg.wParam >> 16) & 0xFFFF) as i16 as f32;
            // delta is usually 120 per click, egui prefers points. We can scale it.
            let points = delta / 120.0 * 50.0;
            raw_input.events.push(Event::Scroll(Vec2::new(0.0, points)));
        }
        win32::WM_KEYDOWN | win32::WM_KEYUP => {
            let pressed = msg.message == win32::WM_KEYDOWN;
            if let Some(key) = translate_key(msg.wParam) {
                raw_input.events.push(Event::Key {
                    key,
                    physical_key: None,
                    pressed,
                    repeat: false, // Could track repeat from lParam, but false is usually fine for simple input
                    modifiers: egui::Modifiers::default(),
                });
            }
        }
        win32::WM_CHAR => {
            let ch = msg.wParam as u8 as char;
            // Filter out control characters
            if !ch.is_control() {
                raw_input.events.push(Event::Text(ch.to_string()));
            }
        }
        _ => {}
    }
}

fn translate_key(vk: usize) -> Option<Key> {
    match vk {
        0x41 => Some(Key::A),
        0x42 => Some(Key::B),
        0x43 => Some(Key::C),
        0x44 => Some(Key::D),
        0x45 => Some(Key::E),
        0x46 => Some(Key::F),
        0x47 => Some(Key::G),
        0x48 => Some(Key::H),
        0x49 => Some(Key::I),
        0x4A => Some(Key::J),
        0x4B => Some(Key::K),
        0x4C => Some(Key::L),
        0x4D => Some(Key::M),
        0x4E => Some(Key::N),
        0x4F => Some(Key::O),
        0x50 => Some(Key::P),
        0x51 => Some(Key::Q),
        0x52 => Some(Key::R),
        0x53 => Some(Key::S),
        0x54 => Some(Key::T),
        0x55 => Some(Key::U),
        0x56 => Some(Key::V),
        0x57 => Some(Key::W),
        0x58 => Some(Key::X),
        0x59 => Some(Key::Y),
        0x5A => Some(Key::Z),
        0x30 => Some(Key::Num0),
        0x31 => Some(Key::Num1),
        0x32 => Some(Key::Num2),
        0x33 => Some(Key::Num3),
        0x34 => Some(Key::Num4),
        0x35 => Some(Key::Num5),
        0x36 => Some(Key::Num6),
        0x37 => Some(Key::Num7),
        0x38 => Some(Key::Num8),
        0x39 => Some(Key::Num9),
        win32::VK_ESCAPE => Some(Key::Escape),
        win32::VK_SPACE => Some(Key::Space),
        win32::VK_TAB => Some(Key::Tab),
        0x0D => Some(Key::Enter),     // VK_RETURN
        0x08 => Some(Key::Backspace), // VK_BACK
        0x2E => Some(Key::Delete),    // VK_DELETE
        0x25 => Some(Key::ArrowLeft),
        0x26 => Some(Key::ArrowUp),
        0x27 => Some(Key::ArrowRight),
        0x28 => Some(Key::ArrowDown),
        _ => None,
    }
}
