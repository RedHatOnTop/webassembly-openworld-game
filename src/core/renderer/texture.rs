//! Texture Management Module
//!
//! Provides texture loading utilities with automatic fallback to generated patterns.
//! Supports creating texture arrays for techniques like triplanar mapping.
//!
//! Features:
//! - Load textures from disk (PNG, JPG, etc.)
//! - Automatic checkerboard fallback if file is missing
//! - TextureArray creation for layered textures
//!
//! Reference: `docs/04_TASKS/task_08_triplanar_texture.md`

use std::path::Path;

// ============================================================================
// Constants
// ============================================================================

/// Default texture size for generated fallback textures.
pub const DEFAULT_TEXTURE_SIZE: u32 = 256;

/// Checkerboard tile size in pixels.
const CHECKER_TILE_SIZE: u32 = 32;

// ============================================================================
// Texture Loading
// ============================================================================

/// Represents loaded texture data ready for GPU upload.
pub struct TextureData {
    /// RGBA pixel data.
    pub data: Vec<u8>,
    /// Texture width in pixels.
    pub width: u32,
    /// Texture height in pixels.
    pub height: u32,
}

impl TextureData {
    /// Loads a texture from disk, or generates a fallback checkerboard pattern.
    ///
    /// # Arguments
    /// * `path` - Path to the texture file.
    /// * `fallback_color1` - First checkerboard color (RGBA).
    /// * `fallback_color2` - Second checkerboard color (RGBA).
    ///
    /// # Returns
    /// `TextureData` containing RGBA pixel data.
    pub fn load_or_fallback(path: &str, fallback_color1: [u8; 4], fallback_color2: [u8; 4]) -> Self {
        if let Ok(data) = Self::load_from_file(path) {
            log::info!("Loaded texture: {}", path);
            data
        } else {
            log::warn!(
                "Failed to load texture '{}', using checkerboard fallback",
                path
            );
            Self::generate_checkerboard(DEFAULT_TEXTURE_SIZE, fallback_color1, fallback_color2)
        }
    }

    /// Attempts to load a texture from a file path.
    fn load_from_file(path: &str) -> Result<Self, image::ImageError> {
        let path = Path::new(path);
        let img = image::open(path)?;
        let rgba = img.to_rgba8();
        let (width, height) = rgba.dimensions();

        Ok(Self {
            data: rgba.into_raw(),
            width,
            height,
        })
    }

    /// Generates a checkerboard pattern texture.
    ///
    /// # Arguments
    /// * `size` - Texture dimensions (square).
    /// * `color1` - First checkerboard color (RGBA).
    /// * `color2` - Second checkerboard color (RGBA).
    pub fn generate_checkerboard(size: u32, color1: [u8; 4], color2: [u8; 4]) -> Self {
        let mut data = Vec::with_capacity((size * size * 4) as usize);

        for y in 0..size {
            for x in 0..size {
                let checker_x = (x / CHECKER_TILE_SIZE) % 2;
                let checker_y = (y / CHECKER_TILE_SIZE) % 2;
                let color = if checker_x == checker_y {
                    color1
                } else {
                    color2
                };
                data.extend_from_slice(&color);
            }
        }

        Self {
            data,
            width: size,
            height: size,
        }
    }

    /// Generates a solid color texture.
    ///
    /// # Arguments
    /// * `size` - Texture dimensions (square).
    /// * `color` - Solid color (RGBA).
    #[allow(dead_code)]
    pub fn generate_solid(size: u32, color: [u8; 4]) -> Self {
        let data = color.repeat((size * size) as usize);
        Self {
            data,
            width: size,
            height: size,
        }
    }
}

// ============================================================================
// Texture Array Builder
// ============================================================================

/// Builder for creating GPU texture arrays from multiple texture layers.
pub struct TextureArrayBuilder {
    /// Collected texture layers.
    layers: Vec<TextureData>,
    /// Target width (all layers must match).
    width: u32,
    /// Target height (all layers must match).
    height: u32,
}

impl TextureArrayBuilder {
    /// Creates a new texture array builder with specified dimensions.
    ///
    /// # Arguments
    /// * `width` - Texture width in pixels.
    /// * `height` - Texture height in pixels.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            layers: Vec::new(),
            width,
            height,
        }
    }

    /// Adds a texture layer to the array.
    ///
    /// If the texture dimensions don't match, it will be resized.
    ///
    /// # Arguments
    /// * `texture` - The texture data to add.
    pub fn add_layer(mut self, texture: TextureData) -> Self {
        // Resize if dimensions don't match
        let resized = if texture.width != self.width || texture.height != self.height {
            log::warn!(
                "Resizing texture from {}x{} to {}x{}",
                texture.width,
                texture.height,
                self.width,
                self.height
            );
            Self::resize_texture(&texture, self.width, self.height)
        } else {
            texture
        };

        self.layers.push(resized);
        self
    }

    /// Resizes a texture to target dimensions using nearest-neighbor sampling.
    fn resize_texture(texture: &TextureData, target_width: u32, target_height: u32) -> TextureData {
        let mut data = Vec::with_capacity((target_width * target_height * 4) as usize);

        for y in 0..target_height {
            for x in 0..target_width {
                // Map target coordinates to source coordinates
                let src_x = (x * texture.width / target_width).min(texture.width - 1);
                let src_y = (y * texture.height / target_height).min(texture.height - 1);

                let src_index = ((src_y * texture.width + src_x) * 4) as usize;
                data.extend_from_slice(&texture.data[src_index..src_index + 4]);
            }
        }

        TextureData {
            data,
            width: target_width,
            height: target_height,
        }
    }

    /// Builds the GPU texture array.
    ///
    /// # Arguments
    /// * `device` - WebGPU device.
    /// * `queue` - WebGPU queue for data upload.
    /// * `label` - Debug label for the texture.
    ///
    /// # Returns
    /// The created texture and its view.
    pub fn build(
        self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        label: &str,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let layer_count = self.layers.len() as u32;

        // Create the texture array
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: layer_count,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Upload each layer
        for (layer_index, layer_data) in self.layers.iter().enumerate() {
            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: layer_index as u32,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                &layer_data.data,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * self.width),
                    rows_per_image: Some(self.height),
                },
                wgpu::Extent3d {
                    width: self.width,
                    height: self.height,
                    depth_or_array_layers: 1,
                },
            );
        }

        // Create view for the entire array
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(&format!("{} View", label)),
            format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: Some(layer_count),
        });

        log::info!(
            "Created texture array '{}': {}x{} x {} layers",
            label,
            self.width,
            self.height,
            layer_count
        );

        (texture, view)
    }
}

// ============================================================================
// Sampler Creation
// ============================================================================

/// Creates a standard terrain sampler with repeat wrapping and linear filtering.
pub fn create_terrain_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Terrain Sampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        lod_min_clamp: 0.0,
        lod_max_clamp: 32.0,
        compare: None,
        anisotropy_clamp: 1,
        border_color: None,
    })
}

// ============================================================================
// Predefined Fallback Colors
// ============================================================================

/// Grass texture fallback colors (green checkerboard).
pub const GRASS_COLOR_1: [u8; 4] = [76, 153, 0, 255]; // Dark green
pub const GRASS_COLOR_2: [u8; 4] = [102, 204, 0, 255]; // Light green

/// Rock texture fallback colors (gray checkerboard).
pub const ROCK_COLOR_1: [u8; 4] = [102, 102, 102, 255]; // Dark gray
pub const ROCK_COLOR_2: [u8; 4] = [153, 153, 153, 255]; // Light gray

/// Snow texture fallback colors (white/light blue checkerboard).
pub const SNOW_COLOR_1: [u8; 4] = [240, 240, 255, 255]; // Near white
pub const SNOW_COLOR_2: [u8; 4] = [220, 230, 255, 255]; // Light blue-white

/// Sand texture fallback colors (yellow/tan checkerboard).
pub const SAND_COLOR_1: [u8; 4] = [194, 178, 128, 255]; // Tan/sand
pub const SAND_COLOR_2: [u8; 4] = [218, 194, 145, 255]; // Light sand
