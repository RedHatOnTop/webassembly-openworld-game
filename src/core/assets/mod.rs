//! Asset Management System
//!
//! Provides robust GLTF loading with automatic fallback to procedural debug meshes.
//! Zero-crash policy: Missing assets display colored placeholders.
//!
//! Reference: docs/04_TASKS/task_13_structures.md

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use std::collections::HashMap;
use std::path::Path;

// ============================================================================
// Structure Manifest - Central Asset Configuration
// ============================================================================

/// Logical structure types mapped to asset filenames.
/// Change these string literals to match your asset filenames.
pub mod manifest {
    pub const TREE_PINE: &str = "assets/models/tree_pine.glb";
    pub const TREE_OAK: &str = "assets/models/tree_oak.glb";
    pub const ROCK_SMALL: &str = "assets/models/rock_small.glb";
    pub const ROCK_LARGE: &str = "assets/models/rock_large.glb";
    pub const RUIN_PILLAR: &str = "assets/models/ruin_pillar.glb";
    pub const RUIN_WALL: &str = "assets/models/ruin_wall.glb";
}

// ============================================================================
// Mesh Vertex Format
// ============================================================================

/// Vertex format for loaded meshes.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub color: [f32; 4],  // Vertex color for fallback meshes
}

impl MeshVertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<MeshVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,  // position
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 12,
                    shader_location: 1,  // normal
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 24,
                    shader_location: 2,  // uv
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 32,
                    shader_location: 3,  // color
                },
            ],
        }
    }
}

// ============================================================================
// Loaded Mesh
// ============================================================================

/// A loaded mesh ready for GPU rendering.
pub struct LoadedMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
    pub is_fallback: bool,  // True if using procedural fallback
}

// ============================================================================
// Fallback Mesh Generation
// ============================================================================

/// Creates a colored cube mesh as fallback when assets are missing.
pub fn create_fallback_cube(device: &wgpu::Device, color: [f32; 4], scale: f32) -> LoadedMesh {
    let half = 0.5 * scale;
    
    // Cube vertices with normals and colors
    let vertices = vec![
        // Front face (+Z)
        MeshVertex { position: [-half, -half, half], normal: [0.0, 0.0, 1.0], uv: [0.0, 1.0], color },
        MeshVertex { position: [half, -half, half], normal: [0.0, 0.0, 1.0], uv: [1.0, 1.0], color },
        MeshVertex { position: [half, half, half], normal: [0.0, 0.0, 1.0], uv: [1.0, 0.0], color },
        MeshVertex { position: [-half, half, half], normal: [0.0, 0.0, 1.0], uv: [0.0, 0.0], color },
        // Back face (-Z)
        MeshVertex { position: [half, -half, -half], normal: [0.0, 0.0, -1.0], uv: [0.0, 1.0], color },
        MeshVertex { position: [-half, -half, -half], normal: [0.0, 0.0, -1.0], uv: [1.0, 1.0], color },
        MeshVertex { position: [-half, half, -half], normal: [0.0, 0.0, -1.0], uv: [1.0, 0.0], color },
        MeshVertex { position: [half, half, -half], normal: [0.0, 0.0, -1.0], uv: [0.0, 0.0], color },
        // Top face (+Y)
        MeshVertex { position: [-half, half, half], normal: [0.0, 1.0, 0.0], uv: [0.0, 1.0], color },
        MeshVertex { position: [half, half, half], normal: [0.0, 1.0, 0.0], uv: [1.0, 1.0], color },
        MeshVertex { position: [half, half, -half], normal: [0.0, 1.0, 0.0], uv: [1.0, 0.0], color },
        MeshVertex { position: [-half, half, -half], normal: [0.0, 1.0, 0.0], uv: [0.0, 0.0], color },
        // Bottom face (-Y)
        MeshVertex { position: [-half, -half, -half], normal: [0.0, -1.0, 0.0], uv: [0.0, 1.0], color },
        MeshVertex { position: [half, -half, -half], normal: [0.0, -1.0, 0.0], uv: [1.0, 1.0], color },
        MeshVertex { position: [half, -half, half], normal: [0.0, -1.0, 0.0], uv: [1.0, 0.0], color },
        MeshVertex { position: [-half, -half, half], normal: [0.0, -1.0, 0.0], uv: [0.0, 0.0], color },
        // Right face (+X)
        MeshVertex { position: [half, -half, half], normal: [1.0, 0.0, 0.0], uv: [0.0, 1.0], color },
        MeshVertex { position: [half, -half, -half], normal: [1.0, 0.0, 0.0], uv: [1.0, 1.0], color },
        MeshVertex { position: [half, half, -half], normal: [1.0, 0.0, 0.0], uv: [1.0, 0.0], color },
        MeshVertex { position: [half, half, half], normal: [1.0, 0.0, 0.0], uv: [0.0, 0.0], color },
        // Left face (-X)
        MeshVertex { position: [-half, -half, -half], normal: [-1.0, 0.0, 0.0], uv: [0.0, 1.0], color },
        MeshVertex { position: [-half, -half, half], normal: [-1.0, 0.0, 0.0], uv: [1.0, 1.0], color },
        MeshVertex { position: [-half, half, half], normal: [-1.0, 0.0, 0.0], uv: [1.0, 0.0], color },
        MeshVertex { position: [-half, half, -half], normal: [-1.0, 0.0, 0.0], uv: [0.0, 0.0], color },
    ];

    let indices: Vec<u32> = vec![
        0, 1, 2, 0, 2, 3,       // Front
        4, 5, 6, 4, 6, 7,       // Back
        8, 9, 10, 8, 10, 11,    // Top
        12, 13, 14, 12, 14, 15, // Bottom
        16, 17, 18, 16, 18, 19, // Right
        20, 21, 22, 20, 22, 23, // Left
    ];

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Fallback Cube Vertex Buffer"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Fallback Cube Index Buffer"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    LoadedMesh {
        vertex_buffer,
        index_buffer,
        index_count: indices.len() as u32,
        is_fallback: true,
    }
}

/// Creates a tree-shaped fallback (tall green cube).
pub fn create_fallback_tree(device: &wgpu::Device) -> LoadedMesh {
    let green = [0.2, 0.6, 0.2, 1.0];
    create_fallback_cube(device, green, 2.0)
}

/// Creates a rock-shaped fallback (gray cube).
pub fn create_fallback_rock(device: &wgpu::Device) -> LoadedMesh {
    let gray = [0.5, 0.5, 0.5, 1.0];
    create_fallback_cube(device, gray, 1.0)
}

/// Creates a ruin-shaped fallback (beige pillar-like cube).
pub fn create_fallback_ruin(device: &wgpu::Device) -> LoadedMesh {
    let beige = [0.8, 0.7, 0.5, 1.0];
    create_fallback_cube(device, beige, 1.5)
}

// ============================================================================
// GLTF Loader with Safe Fallback
// ============================================================================

/// Safely loads a GLTF/GLB file, returning a fallback mesh on failure.
/// This ensures the engine NEVER crashes due to missing assets.
pub fn load_gltf_safe(
    device: &wgpu::Device,
    path: &str,
    fallback: LoadedMesh,
) -> LoadedMesh {
    match load_gltf_internal(device, path) {
        Ok(mesh) => {
            log::info!("Loaded asset: {}", path);
            mesh
        }
        Err(e) => {
            log::warn!("Asset '{}' missing or invalid ({}), using fallback", path, e);
            fallback
        }
    }
}

/// Loads a GLTF/GLB file for drag-drop debug testing.
/// Returns a magenta fallback cube if loading fails (easy to spot errors).
/// Used by the Debug GUI system for runtime asset testing.
pub fn load_gltf_debug(device: &wgpu::Device, path: &str) -> LoadedMesh {
    match load_gltf_internal(device, path) {
        Ok(mesh) => {
            log::info!("Debug: Loaded asset: {}", path);
            mesh
        }
        Err(e) => {
            log::warn!("Debug: Asset '{}' failed to load: {}", path, e);
            // Magenta fallback - highly visible to indicate error
            create_fallback_cube(device, [1.0, 0.0, 1.0, 1.0], 1.0)
        }
    }
}

/// Internal GLTF loading implementation.
fn load_gltf_internal(device: &wgpu::Device, path: &str) -> Result<LoadedMesh, String> {
    // Check if file exists
    if !Path::new(path).exists() {
        return Err("File not found".to_string());
    }

    // Load GLTF
    let (document, buffers, _images) = gltf::import(path)
        .map_err(|e| format!("GLTF parse error: {}", e))?;

    // Get first mesh
    let mesh = document.meshes().next()
        .ok_or("No meshes in GLTF file")?;

    // Get first primitive
    let primitive = mesh.primitives().next()
        .ok_or("No primitives in mesh")?;

    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

    // Read positions
    let positions: Vec<[f32; 3]> = reader.read_positions()
        .ok_or("No positions in mesh")?
        .collect();

    // Read normals (or generate default)
    let normals: Vec<[f32; 3]> = reader.read_normals()
        .map(|n| n.collect())
        .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);

    // Read UVs (or generate default)
    let uvs: Vec<[f32; 2]> = reader.read_tex_coords(0)
        .map(|t| t.into_f32().collect())
        .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);

    // Read vertex colors (or use white)
    let colors: Vec<[f32; 4]> = reader.read_colors(0)
        .map(|c| c.into_rgba_f32().collect())
        .unwrap_or_else(|| vec![[1.0, 1.0, 1.0, 1.0]; positions.len()]);

    // Combine into vertices
    let vertices: Vec<MeshVertex> = positions.iter().enumerate().map(|(i, &pos)| {
        MeshVertex {
            position: pos,
            normal: normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]),
            uv: uvs.get(i).copied().unwrap_or([0.0, 0.0]),
            color: colors.get(i).copied().unwrap_or([1.0, 1.0, 1.0, 1.0]),
        }
    }).collect();

    // Read indices
    let indices: Vec<u32> = reader.read_indices()
        .ok_or("No indices in mesh")?
        .into_u32()
        .collect();

    // Create GPU buffers
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("Mesh Vertex Buffer: {}", path)),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("Mesh Index Buffer: {}", path)),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    Ok(LoadedMesh {
        vertex_buffer,
        index_buffer,
        index_count: indices.len() as u32,
        is_fallback: false,
    })
}

// ============================================================================
// Structure Types
// ============================================================================

/// Types of structures that can be placed in the world.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StructureType {
    TreePine,
    TreeOak,
    RockSmall,
    RockLarge,
    RuinPillar,
    RuinWall,
}

impl StructureType {
    /// Returns the asset path for this structure type.
    pub fn asset_path(&self) -> &'static str {
        match self {
            StructureType::TreePine => manifest::TREE_PINE,
            StructureType::TreeOak => manifest::TREE_OAK,
            StructureType::RockSmall => manifest::ROCK_SMALL,
            StructureType::RockLarge => manifest::ROCK_LARGE,
            StructureType::RuinPillar => manifest::RUIN_PILLAR,
            StructureType::RuinWall => manifest::RUIN_WALL,
        }
    }
}

// ============================================================================
// Model Manager
// ============================================================================

/// Manages all loaded structure models with automatic fallback.
pub struct ModelManager {
    models: HashMap<StructureType, LoadedMesh>,
}

impl ModelManager {
    /// Creates a new ModelManager and loads all structure models.
    pub fn new(device: &wgpu::Device) -> Self {
        let mut models = HashMap::new();

        // Load trees
        models.insert(
            StructureType::TreePine,
            load_gltf_safe(device, manifest::TREE_PINE, create_fallback_tree(device)),
        );
        models.insert(
            StructureType::TreeOak,
            load_gltf_safe(device, manifest::TREE_OAK, create_fallback_tree(device)),
        );

        // Load rocks
        models.insert(
            StructureType::RockSmall,
            load_gltf_safe(device, manifest::ROCK_SMALL, create_fallback_rock(device)),
        );
        models.insert(
            StructureType::RockLarge,
            load_gltf_safe(device, manifest::ROCK_LARGE, create_fallback_rock(device)),
        );

        // Load ruins
        models.insert(
            StructureType::RuinPillar,
            load_gltf_safe(device, manifest::RUIN_PILLAR, create_fallback_ruin(device)),
        );
        models.insert(
            StructureType::RuinWall,
            load_gltf_safe(device, manifest::RUIN_WALL, create_fallback_ruin(device)),
        );

        log::info!("ModelManager loaded {} structure types", models.len());

        Self { models }
    }

    /// Gets a loaded mesh by structure type.
    pub fn get(&self, structure_type: StructureType) -> &LoadedMesh {
        self.models.get(&structure_type).expect("All structure types should be loaded")
    }

    /// Returns all loaded structure types and their meshes.
    pub fn iter(&self) -> impl Iterator<Item = (&StructureType, &LoadedMesh)> {
        self.models.iter()
    }
}
