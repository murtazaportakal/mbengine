//! Abstract Render Device interface.

/// Core trait representing the rendering backend.
pub trait RenderDevice {
    /// Wait for the device to finish all operations.
    fn wait_idle(&self);
    
    /// Shut down and clean up device resources.
    fn shutdown(&mut self);
}
