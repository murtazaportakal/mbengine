use rodio::{OutputStream, OutputStreamHandle, SpatialSink};

use crate::ecs::types::ComponentMask;
use crate::ecs::{System, World};
use crate::math::vec::Vec3;

pub struct AudioSubsystem {
    _stream: OutputStream,
    stream_handle: OutputStreamHandle,
}

impl AudioSubsystem {
    pub fn new() -> Option<Self> {
        let (_stream, stream_handle) = OutputStream::try_default().ok()?;
        Some(Self {
            _stream,
            stream_handle,
        })
    }

    pub fn handle(&self) -> &OutputStreamHandle {
        &self.stream_handle
    }
}

/// Initial capacity for the sink array. Covers up to 128 simultaneous
/// audio emitters without any reallocation. Growth beyond this only
/// happens at entity creation time, never on the per-frame update path.
const INITIAL_SINK_CAPACITY: usize = 128;

pub struct AudioSystem {
    /// Flat array of spatial sinks, indexed directly by entity index.
    /// `None` = no active sink for that slot. Pre-allocated to
    /// `INITIAL_SINK_CAPACITY` to avoid hot-path heap allocations.
    sinks: Vec<Option<SpatialSink>>,
    stream_handle: Option<OutputStreamHandle>,
    vfs: crate::vfs::Vfs,
}

impl AudioSystem {
    pub fn new(subsystem: Option<&AudioSubsystem>, vfs: &crate::vfs::Vfs) -> Self {
        let mut sinks = Vec::with_capacity(INITIAL_SINK_CAPACITY);
        sinks.resize_with(INITIAL_SINK_CAPACITY, || None);
        Self {
            sinks,
            stream_handle: subsystem.map(|s| s.handle().clone()),
            vfs: vfs.clone(),
        }
    }

    /// Ensure the sink array can hold an entry at `index`.
    /// Only called when a new emitter entity is encountered, not per-frame.
    #[inline]
    fn ensure_capacity(&mut self, index: usize) {
        if index >= self.sinks.len() {
            self.sinks.resize_with(index + 1, || None);
        }
    }
}

impl System for AudioSystem {
    fn read_components(&self) -> ComponentMask {
        crate::ecs::types::build_mask(&[
            crate::ecs::types::get_component_type_id::<
                crate::ecs::components::AudioListenerComponent,
            >(),
            crate::ecs::types::get_component_type_id::<crate::ecs::components::AudioEmitterComponent>(
            ),
            crate::ecs::types::get_component_type_id::<crate::ecs::components::TransformComponent>(
            ),
        ])
    }

    fn write_components(&self) -> ComponentMask {
        0
    }

    fn update(&mut self, _dt: f32, world: &World) {
        // Find listener position
        let mut listener_pos = Vec3::new(0.0, 0.0, 0.0);

        let listener_array =
            world.get_component_array::<crate::ecs::components::AudioListenerComponent>();
        let transform_array =
            world.get_component_array::<crate::ecs::components::TransformComponent>();

        for &entity_index in listener_array.dense_entities_slice() {
            if transform_array.has(entity_index) {
                let t = unsafe { transform_array.get(entity_index) };
                listener_pos = t.position;
                break;
            }
        }

        // Extremely naive left/right ear positions for SpatialSink
        let left_ear = [listener_pos.x - 1.0, listener_pos.y, listener_pos.z];
        let right_ear = [listener_pos.x + 1.0, listener_pos.y, listener_pos.z];

        let emitter_array =
            world.get_component_array::<crate::ecs::components::AudioEmitterComponent>();

        for &entity_index in emitter_array.dense_entities_slice() {
            if transform_array.has(entity_index) {
                let emitter = unsafe { emitter_array.get(entity_index) };
                let transform = unsafe { transform_array.get(entity_index) };

                let emitter_pos = [
                    transform.position.x,
                    transform.position.y,
                    transform.position.z,
                ];

                let idx = entity_index as usize;
                self.ensure_capacity(idx);

                if let Some(stream_handle) = &self.stream_handle {
                    // Create sink if it doesn't exist
                    let sink = self.sinks[idx].get_or_insert_with(|| {
                        let s =
                            SpatialSink::try_new(stream_handle, emitter_pos, left_ear, right_ear)
                                .unwrap();

                        // Load and play the audio file via VFS
                        let path_str = emitter.asset_path.as_str();
                        if let Ok(bytes) = self.vfs.read_bytes(path_str) {
                            let cursor = std::io::Cursor::new(bytes);
                            if let Ok(source) = rodio::Decoder::new(cursor) {
                                if emitter.loop_audio {
                                    s.append(rodio::Source::repeat_infinite(source));
                                } else {
                                    s.append(source);
                                }
                            }
                        }

                        if !emitter.is_playing {
                            s.pause();
                        }
                        s
                    });

                    // Sync properties
                    sink.set_emitter_position(emitter_pos);
                    sink.set_left_ear_position(left_ear);
                    sink.set_right_ear_position(right_ear);

                    // Adjust volume linearly by distance
                    let dx = transform.position.x - listener_pos.x;
                    let dy = transform.position.y - listener_pos.y;
                    let dz = transform.position.z - listener_pos.z;
                    let distance = (dx * dx + dy * dy + dz * dz).sqrt();

                    let mut vol = 0.0;
                    if distance < emitter.max_distance {
                        vol = 1.0 - (distance / emitter.max_distance);
                    }
                    vol *= emitter.volume;

                    sink.set_volume(vol);

                    if emitter.is_playing && sink.is_paused() {
                        sink.play();
                    } else if !emitter.is_playing && !sink.is_paused() {
                        sink.pause();
                    }
                }
            }
        }

        // Clean up sinks for deleted emitters — linear scan over occupied slots.
        // No allocation: we just set unused slots to None.
        for i in 0..self.sinks.len() {
            if self.sinks[i].is_some() && !emitter_array.has(i as u32) {
                self.sinks[i] = None;
            }
        }
    }
}
