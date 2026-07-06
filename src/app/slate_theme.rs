use egui::{Color32, Rounding, Stroke, Style, Visuals, Margin, Vec2, epaint::Shadow};

pub const COLOR_BG_GLOBAL: Color32 = Color32::from_rgb(18, 18, 18);
pub const COLOR_BG_PANEL: Color32 = Color32::from_rgb(30, 30, 30);
pub const COLOR_BG_EXTREME: Color32 = Color32::from_rgb(10, 10, 10);
pub const COLOR_STROKE: Color32 = Color32::from_rgb(51, 51, 51);
pub const COLOR_WIDGET_INACTIVE: Color32 = Color32::from_rgb(37, 37, 37);
pub const COLOR_WIDGET_HOVERED: Color32 = Color32::from_rgb(50, 50, 50);
pub const COLOR_WIDGET_ACTIVE: Color32 = Color32::from_rgb(70, 70, 70);
pub const COLOR_ACCENT: Color32 = Color32::from_rgb(0, 112, 224);
pub const COLOR_TEXT_HOVERED: Color32 = Color32::WHITE;

pub struct EngineTheme {
    pub bg_global: Color32,
    pub bg_panel: Color32,
    pub bg_extreme: Color32,
    pub stroke: Color32,
    pub widget_inactive: Color32,
    pub widget_hovered: Color32,
    pub widget_active: Color32,
    pub accent: Color32,
    pub text_hovered: Color32,
    pub text_normal: Color32,
}

impl EngineTheme {
    pub fn slate() -> Self {
        Self {
            bg_global: COLOR_BG_GLOBAL,
            bg_panel: COLOR_BG_PANEL,
            bg_extreme: COLOR_BG_EXTREME,
            stroke: COLOR_STROKE,
            widget_inactive: COLOR_WIDGET_INACTIVE,
            widget_hovered: COLOR_WIDGET_HOVERED,
            widget_active: COLOR_WIDGET_ACTIVE,
            accent: Color32::from_rgb(20, 70, 120),
            text_hovered: COLOR_TEXT_HOVERED,
            text_normal: Color32::from_rgb(200, 200, 200),
        }
    }
}

pub fn apply_slate_theme(ctx: &egui::Context) {
    let mut style = Style::default();
    
    // 1. High-Density Spacing
    style.spacing.item_spacing = Vec2::new(4.0, 4.0); 
    style.spacing.window_margin = Margin::same(6.0);
    style.spacing.button_padding = Vec2::new(8.0, 2.0);
    style.spacing.indent = 12.0;
    
    // 2. The Zero-Rounding Rule (Strict 90-degree corners)
    style.visuals.window_rounding = Rounding::same(0.0);
    style.visuals.menu_rounding = Rounding::same(0.0);
    style.visuals.widgets.noninteractive.rounding = Rounding::same(0.0);
    style.visuals.widgets.inactive.rounding = Rounding::same(0.0);
    style.visuals.widgets.hovered.rounding = Rounding::same(0.0);
    style.visuals.widgets.active.rounding = Rounding::same(0.0);
    style.visuals.widgets.open.rounding = Rounding::same(0.0);

    // 3. Monochromatic Charcoal Step-Scale
    let mut visuals = Visuals::dark();
    
    // Background fills
    visuals.window_fill = Color32::from_rgb(18, 18, 18);      // Deep global background (#121212)
    visuals.panel_fill = Color32::from_rgb(30, 30, 30);       // Tool panel background (#1E1E1E)
    visuals.extreme_bg_color = Color32::from_rgb(10, 10, 10); // Inputs, text areas (#0A0A0A)
    
    // 1px Structural Borders (No Drop Shadows)
    visuals.window_stroke = Stroke::new(1.0, Color32::from_rgb(51, 51, 51)); // #333333
    visuals.window_shadow = Shadow::NONE;
    visuals.popup_shadow = Shadow::NONE;
    
    // 4. Interactive Widget States
    // Inactive / Idle Buttons
    visuals.widgets.inactive.bg_fill = Color32::from_rgb(37, 37, 37); // #252525
    visuals.widgets.inactive.bg_stroke = Stroke::new(1.0, Color32::from_rgb(51, 51, 51));
    visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, Color32::from_rgb(200, 200, 200));
    
    // Hovered 
    visuals.widgets.hovered.bg_fill = Color32::from_rgb(50, 50, 50); // #323232
    visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, Color32::from_rgb(80, 80, 80));
    visuals.widgets.hovered.fg_stroke = Stroke::new(1.0, Color32::from_rgb(255, 255, 255)); // Crisp white text
    
    // Active / Clicked
    visuals.widgets.active.bg_fill = Color32::from_rgb(70, 70, 70); // #464646
    visuals.widgets.active.bg_stroke = Stroke::new(1.0, Color32::from_rgb(110, 110, 110));
    visuals.widgets.active.fg_stroke = Stroke::new(1.0, Color32::from_rgb(255, 255, 255));
    
    // Selection Accent (The Unreal Engine Blue Accent)
    visuals.selection.bg_fill = Color32::from_rgb(0, 112, 224); // #0070E0
    visuals.selection.stroke = Stroke::new(1.0, Color32::from_rgb(255, 255, 255));

    style.visuals = visuals;
    ctx.set_style(style);
}

pub struct EditorFrame;

impl EditorFrame {
    /// Standard panel frame (Left, Right, Bottom, Top)
    pub fn panel() -> egui::Frame {
        egui::Frame::default()
            .fill(COLOR_BG_PANEL)
            .stroke(Stroke::new(1.0, COLOR_STROKE))
            .inner_margin(Margin::same(6.0))
    }

    /// Central panel frame for the Viewport (flush, no margins)
    pub fn central_viewport() -> egui::Frame {
        egui::Frame::default()
            .fill(COLOR_BG_GLOBAL)
            .stroke(Stroke::NONE)
            .inner_margin(Margin::same(0.0))
    }
}
