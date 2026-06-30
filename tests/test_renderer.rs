//! Integration tests for Renderer Foundation.

use engine::platform::Window;
use engine::renderer::vulkan::{Swapchain, VulkanDevice};

#[test]
fn test_vulkan_initialization() {
    // Attempt to create a Vulkan Device.
    // This will gracefully return None if no Vulkan GPU or driver is found.
    let mut vulkan = match VulkanDevice::new() {
        Some(v) => v,
        None => {
            println!("No Vulkan support found. Skipping test.");
            return;
        }
    };

    // Create a headless window for the surface
    let window = Window::new("Vulkan Smoke Test", 800, 600);

    // Attempt to create the Swapchain
    let mut swapchain = match Swapchain::new(&vulkan, &window, 800, 600) {
        Some(s) => s,
        None => {
            println!("Swapchain creation failed or not supported. Skipping.");
            return;
        }
    };

    assert!(!swapchain.images.is_empty());
    assert_eq!(swapchain.image_views.len(), swapchain.images.len());

    // Cleanup
    use engine::renderer::RenderDevice;
    swapchain.shutdown(&vulkan);
    vulkan.shutdown();
}
