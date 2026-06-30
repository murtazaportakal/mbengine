//! Win32 Window creation and message loop.
//!
//! Exposes a raw `HWND` for the renderer to attach to later.

use crate::platform::win32;
use std::ptr;

/// A Win32 window.
pub struct Window {
    hwnd: win32::HWND,
    hinstance: win32::HINSTANCE,
    should_close: bool,
    pub width: u32,
    pub height: u32,
    resized: bool,
}

impl Window {
    /// Create a new window with the given title and dimensions.
    pub fn new(title: &str, width: i32, height: i32) -> Self {
        unsafe {
            let hinstance = win32::GetModuleHandleA(ptr::null());

            let class_name = b"EngineWindowClass\0".as_ptr();
            let window_name = format!("{}\0", title);

            // Set up the window class
            let mut wnd_class: win32::WNDCLASSEXA = std::mem::zeroed();
            wnd_class.cbSize = std::mem::size_of::<win32::WNDCLASSEXA>() as win32::UINT;
            wnd_class.style = win32::CS_HREDRAW | win32::CS_VREDRAW | win32::CS_OWNDC;
            wnd_class.lpfnWndProc = Some(window_proc);
            wnd_class.hInstance = hinstance;
            wnd_class.lpszClassName = class_name;

            let atom = win32::RegisterClassExA(&wnd_class);
            assert!(atom != 0, "Failed to register window class.");

            // Create the window
            let hwnd = win32::CreateWindowExA(
                0,
                class_name,
                window_name.as_ptr(),
                win32::WS_OVERLAPPEDWINDOW,
                win32::CW_USEDEFAULT,
                win32::CW_USEDEFAULT,
                width,
                height,
                ptr::null_mut(),
                ptr::null_mut(),
                hinstance,
                ptr::null_mut(),
            );

            assert!(!hwnd.is_null(), "Failed to create window.");

            win32::ShowWindow(hwnd, win32::SW_SHOW);

            let window = Self {
                hwnd,
                hinstance,
                should_close: false,
                width: width as u32,
                height: height as u32,
                resized: false,
            };

            window
        }
    }

    /// Pump messages and update input state. Returns `false` if the window should close.
    pub fn poll_events(&mut self, input: &mut crate::app::Input) -> bool {
        input.update();

        let mut msg: win32::MSG = unsafe { std::mem::zeroed() };
        
        while unsafe { win32::PeekMessageA(&mut msg, ptr::null_mut(), 0, 0, win32::PM_REMOVE) } != 0 {
            if msg.message == win32::WM_QUIT {
                self.should_close = true;
                return false;
            }

            match msg.message {
                win32::WM_KEYDOWN => {
                    let vk = msg.wParam;
                    if vk < 256 {
                        input.keys[vk] = true;
                    }
                }
                win32::WM_KEYUP => {
                    let vk = msg.wParam;
                    if vk < 256 {
                        input.keys[vk] = false;
                    }
                }
                win32::WM_MOUSEMOVE => {
                    let x = (msg.lParam & 0xFFFF) as i16 as i32;
                    let y = ((msg.lParam >> 16) & 0xFFFF) as i16 as i32;
                    
                    if input.first_mouse {
                        input.mouse_x = x;
                        input.mouse_y = y;
                        input.first_mouse = false;
                    }
                    
                    input.mouse_dx += x - input.mouse_x;
                    input.mouse_dy += y - input.mouse_y;
                    input.mouse_x = x;
                    input.mouse_y = y;
                }
                win32::WM_LBUTTONDOWN => input.keys[win32::VK_LBUTTON] = true,
                win32::WM_LBUTTONUP => input.keys[win32::VK_LBUTTON] = false,
                win32::WM_RBUTTONDOWN => input.keys[win32::VK_RBUTTON] = true,
                win32::WM_RBUTTONUP => input.keys[win32::VK_RBUTTON] = false,
                win32::WM_MBUTTONDOWN => input.keys[win32::VK_MBUTTON] = true,
                win32::WM_MBUTTONUP => input.keys[win32::VK_MBUTTON] = false,
                _ => {}
            }
            
            crate::app::egui_win32::translate_win32_to_egui(&msg, &mut input.egui_input);
            
            unsafe {
                win32::TranslateMessage(&msg);
                win32::DispatchMessageA(&msg);
            }
        }
        
        !self.should_close
    }

    /// Raw HWND handle, needed for Vulkan surface creation.
    pub fn hwnd(&self) -> win32::HWND {
        self.hwnd
    }

    /// Raw HINSTANCE handle, needed for Vulkan surface creation.
    pub fn hinstance(&self) -> win32::HINSTANCE {
        self.hinstance
    }

    /// Check if the window was resized since the last check.
    pub fn check_and_clear_resized(&mut self) -> bool {
        let mut rect = win32::RECT { left: 0, top: 0, right: 0, bottom: 0 };
        unsafe {
            win32::GetClientRect(self.hwnd, &mut rect);
        }
        let w = (rect.right - rect.left) as u32;
        let h = (rect.bottom - rect.top) as u32;

        if w != self.width || h != self.height {
            self.width = w;
            self.height = h;
            self.resized = true;
        }

        if self.resized {
            self.resized = false;
            true
        } else {
            false
        }
    }
}

/// Global Win32 message callback.
unsafe extern "system" fn window_proc(
    hwnd: win32::HWND,
    msg: win32::UINT,
    wparam: win32::WPARAM,
    lparam: win32::LPARAM,
) -> win32::LRESULT {
    match msg {
        win32::WM_CLOSE | win32::WM_DESTROY => {
            win32::PostQuitMessage(0);
            0
        }
        _ => win32::DefWindowProcA(hwnd, msg, wparam, lparam),
    }
}
