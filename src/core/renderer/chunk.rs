//! Chunk Module
//!
//! Defines the Chunk structure for GPU-driven terrain tiles.
//! Part of the Multi-Chunk Management System (Task 10).
//!
//! Key Design:
//! - Pre-allocated vertex buffers (no runtime allocation)
//! - Chunk coordinates are logical grid positions, not world positions
//! - World position = chunk_coord * CHUNK_SIZE

use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

// ============================================================================
// Constants
// ============================================================================

/// Size of each chunk in world units (100x100 vertex grid with 1 unit spacing).
pub const CHUNK_SIZE: f32 = 100.0;

/// Render distance in chunks from camera (3 = 7x7 grid around player).
pub const RENDER_DISTANCE: i32 = 3;

/// Total number of chunks in the pool.
/// Diameter = 2 * RENDER_DISTANCE + 1 = 7
/// Total = 7 * 7 = 49
pub const CHUNK_POOL_SIZE: usize = ((RENDER_DISTANCE * 2 + 1) * (RENDER_DISTANCE * 2 + 1)) as usize;

/// Grid dimension per chunk (matches terrain.rs GRID_SIZE).
pub const GRID_SIZE: u32 = 100;

/// Vertices per chunk.
pub const VERTICES_PER_CHUNK: u32 = GRID_SIZE * GRID_SIZE;

/// Bytes per vertex (position: vec4, normal: vec4 = 32 bytes).
pub const VERTEX_SIZE: u64 = 32;

// ============================================================================
// Chunk Uniforms
// ============================================================================

/// Per-chunk uniform data sent to compute shader.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ChunkUniforms {
    /// Chunk offset in world space (world_x = chunk_coord.0 * CHUNK_SIZE).
    pub offset_x: f32,
    pub offset_z: f32,
    /// Grid size for compute shader.
    pub grid_size: u32,
    /// Padding for 16-byte alignment.
    pub _padding: u32,
}

impl ChunkUniforms {
    pub fn new(coord: (i32, i32)) -> Self {
        Self {
            offset_x: coord.0 as f32 * CHUNK_SIZE,
            offset_z: coord.1 as f32 * CHUNK_SIZE,
            grid_size: GRID_SIZE,
            _padding: 0,
        }
    }

    pub fn update(&mut self, coord: (i32, i32)) {
        self.offset_x = coord.0 as f32 * CHUNK_SIZE;
        self.offset_z = coord.1 as f32 * CHUNK_SIZE;
    }
}

// Compile-time size verification (must be 16 bytes, 16-byte aligned)
const _: () = assert!(std::mem::size_of::<ChunkUniforms>() == 16);

// ============================================================================
// Chunk Structure
// ============================================================================

/// A single terrain chunk with pre-allocated GPU resources.
///
/// Each chunk represents a GRID_SIZE x GRID_SIZE section of the world.
/// Chunks are recycled (not deallocated) when the camera moves.
pub struct Chunk {
    /// Logical chunk coordinate (NOT world position).
    /// World position = coord * CHUNK_SIZE.
    pub coord: (i32, i32),

    /// Pre-allocated vertex storage buffer.
    /// Size: VERTICES_PER_CHUNK * VERTEX_SIZE bytes.
    pub vertex_buffer: wgpu::Buffer,

    /// Uniform buffer for this chunk (contains offset).
    pub uniform_buffer: wgpu::Buffer,

    /// Current uniform values.
    pub uniforms: ChunkUniforms,

    /// Bind group for compute shader (vertex buffer + uniforms).
    pub compute_bind_group: wgpu::BindGroup,

    /// Bind group for render shader (vertex buffer as storage).
    pub render_bind_group: wgpu::BindGroup,

    /// Whether this chunk should be rendered.
    pub active: bool,

    /// Whether this chunk needs its terrain data regenerated.
    pub dirty: bool,
}

impl Chunk {
    /// Creates a new chunk with pre-allocated GPU resources.
    ///
    /// # Arguments
    /// * `device` - WebGPU device for buffer creation.
    /// * `coord` - Initial logical chunk coordinate.
    /// * `compute_storage_layout` - Bind group layout for compute shader.
    /// * `render_storage_layout` - Bind group layout for render shader.
    pub fn new(
        device: &wgpu::Device,
        coord: (i32, i32),
        compute_storage_layout: &wgpu::BindGroupLayout,
        render_storage_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        // Pre-allocate vertex buffer
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("Chunk Vertex Buffer ({}, {})", coord.0, coord.1)),
            size: (VERTICES_PER_CHUNK as u64) * VERTEX_SIZE,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create uniform buffer
        let uniforms = ChunkUniforms::new(coord);
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("Chunk Uniform Buffer ({}, {})", coord.0, coord.1)),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create compute bind group (vertex buffer read_write + uniforms)
        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Chunk Compute Bind Group ({}, {})", coord.0, coord.1)),
            layout: compute_storage_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        // Create render bind group (vertex buffer read-only)
        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Chunk Render Bind Group ({}, {})", coord.0, coord.1)),
            layout: render_storage_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vertex_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            coord,
            vertex_buffer,
            uniform_buffer,
            uniforms,
            compute_bind_group,
            render_bind_group,
            active: true,
            dirty: true, // New chunks need initial generation
        }
    }

    /// Updates the chunk's coordinate (for recycling).
    /// Marks the chunk as dirty so terrain will be regenerated.
    pub fn set_coord(&mut self, queue: &wgpu::Queue, new_coord: (i32, i32)) {
        self.coord = new_coord;
        self.uniforms.update(new_coord);
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[self.uniforms]));
        self.dirty = true;
        self.active = true;
    }

    /// Returns the world-space center position of this chunk.
    pub fn world_center(&self) -> (f32, f32) {
        (
            self.coord.0 as f32 * CHUNK_SIZE + CHUNK_SIZE * 0.5,
            self.coord.1 as f32 * CHUNK_SIZE + CHUNK_SIZE * 0.5,
        )
    }
}

// ============================================================================
// Chunk Pool Management
// ============================================================================

/// Calculates which chunk coordinate contains a world position.
pub fn world_to_chunk_coord(world_x: f32, world_z: f32) -> (i32, i32) {
    (
        (world_x / CHUNK_SIZE).floor() as i32,
        (world_z / CHUNK_SIZE).floor() as i32,
    )
}

/// Generates the set of chunk coordinates that should be loaded.
/// Returns a 7x7 grid of coordinates centered on `center`.
pub fn get_required_chunks(center: (i32, i32)) -> Vec<(i32, i32)> {
    let mut coords = Vec::with_capacity(CHUNK_POOL_SIZE);
    for dz in -RENDER_DISTANCE..=RENDER_DISTANCE {
        for dx in -RENDER_DISTANCE..=RENDER_DISTANCE {
            coords.push((center.0 + dx, center.1 + dz));
        }
    }
    coords
}

/// Checks if a chunk coordinate is within render distance of the center.
pub fn is_chunk_in_range(chunk: (i32, i32), center: (i32, i32)) -> bool {
    let dx = (chunk.0 - center.0).abs();
    let dz = (chunk.1 - center.1).abs();
    dx <= RENDER_DISTANCE && dz <= RENDER_DISTANCE
}
