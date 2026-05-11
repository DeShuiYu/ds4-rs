//! GGUF v3 file format parser for the DS4 inference engine.
//!
//! Maps to the GGUF parsing section from `ds4.c` (lines ~800-1800).
//!
//! The loader mmap's the model once, records metadata KV pairs and tensor
//! descriptors, and leaves the tensor payload bytes in place.  Inference code
//! accesses weights through offset-addressed slices of the mapping without
//! copying the GGUF into private structures.
//!
//! GGUF v3 layout (all little-endian):
//!
//!   [0x00] magic       u32  = 0x46554747 ("GGUF")
//!   [0x04] version     u32  = 3
//!   [0x08] n_tensors   u64
//!   [0x10] n_kv        u64
//!   [0x18] metadata KV pairs  (key: string, value: typed value)
//!   [...]  tensor infos       (name, ndim, dims[], type, rel_offset)
//!   [align] tensor data       (referenced by abs_offset = tensor_data_pos + rel_offset)

use std::fs::File;
use std::path::Path;

use anyhow::{bail, Context, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::Mmap;

use crate::types::TensorType;

// ── Constants ─────────────────────────────────────────────────────────────

/// GGUF magic bytes: "GGUF" in little-endian u32.
pub const DS4_GGUF_MAGIC: u32 = 0x46554747;

/// Maximum number of tensor dimensions supported by the GGUF spec.
pub const DS4_MAX_DIMS: usize = 8;

// ── Metadata value types ──────────────────────────────────────────────────

/// GGUF metadata value type tags.
///
/// These match the `GGUF_VALUE_*` constants in `ds4.c` lines 814-828.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MetadataType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl MetadataType {
    /// Look up a `MetadataType` from its raw GGUF tag value.
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(MetadataType::Uint8),
            1 => Some(MetadataType::Int8),
            2 => Some(MetadataType::Uint16),
            3 => Some(MetadataType::Int16),
            4 => Some(MetadataType::Uint32),
            5 => Some(MetadataType::Int32),
            6 => Some(MetadataType::Float32),
            7 => Some(MetadataType::Bool),
            8 => Some(MetadataType::String),
            9 => Some(MetadataType::Array),
            10 => Some(MetadataType::Uint64),
            11 => Some(MetadataType::Int64),
            12 => Some(MetadataType::Float64),
            _ => None,
        }
    }

    /// Return the size in bytes of a scalar value of this type, or `None` for
    /// compound types (string, array) that require variable-length decoding.
    pub fn scalar_size(&self) -> Option<u64> {
        match self {
            MetadataType::Uint8 | MetadataType::Int8 | MetadataType::Bool => Some(1),
            MetadataType::Uint16 | MetadataType::Int16 => Some(2),
            MetadataType::Uint32 | MetadataType::Int32 | MetadataType::Float32 => Some(4),
            MetadataType::Uint64 | MetadataType::Int64 | MetadataType::Float64 => Some(8),
            MetadataType::String | MetadataType::Array => None,
        }
    }
}

/// Runtime representation of a parsed GGUF metadata value.
///
/// For `Array`, the items are stored as a `Vec<MetadataValue>` and all share
/// the same item type.  For scalar types the value is stored inline.
#[derive(Debug, Clone)]
pub enum MetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl MetadataValue {
    /// Return the `MetadataType` tag for this value.
    pub fn value_type(&self) -> MetadataType {
        match self {
            MetadataValue::Uint8(_) => MetadataType::Uint8,
            MetadataValue::Int8(_) => MetadataType::Int8,
            MetadataValue::Uint16(_) => MetadataType::Uint16,
            MetadataValue::Int16(_) => MetadataType::Int16,
            MetadataValue::Uint32(_) => MetadataType::Uint32,
            MetadataValue::Int32(_) => MetadataType::Int32,
            MetadataValue::Float32(_) => MetadataType::Float32,
            MetadataValue::Bool(_) => MetadataType::Bool,
            MetadataValue::String(_) => MetadataType::String,
            MetadataValue::Array(_) => MetadataType::Array,
            MetadataValue::Uint64(_) => MetadataType::Uint64,
            MetadataValue::Int64(_) => MetadataType::Int64,
            MetadataValue::Float64(_) => MetadataType::Float64,
        }
    }

    // ── Conversion helpers ────────────────────────────────────────────

    pub fn as_u32(&self) -> Option<u32> {
        match self {
            MetadataValue::Uint32(v) => Some(*v),
            MetadataValue::Uint64(v) if *v <= u64::from(u32::MAX) => Some(*v as u32),
            MetadataValue::Int32(v) if *v >= 0 => Some(*v as u32),
            MetadataValue::Int64(v) if *v >= 0 && *v <= i64::from(u32::MAX) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            MetadataValue::Float32(v) => Some(*v),
            MetadataValue::Float64(v) => Some(*v as f32),
            MetadataValue::Uint32(v) => Some(*v as f32),
            MetadataValue::Int32(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            MetadataValue::Bool(v) => Some(*v),
            MetadataValue::Uint8(v) => Some(*v != 0),
            MetadataValue::Int8(v) => Some(*v != 0),
            _ => None,
        }
    }

    pub fn as_string(&self) -> Option<&str> {
        match self {
            MetadataValue::String(v) => Some(v.as_str()),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            MetadataValue::Uint64(v) => Some(*v),
            MetadataValue::Uint32(v) => Some(u64::from(*v)),
            MetadataValue::Int64(v) if *v >= 0 => Some(*v as u64),
            MetadataValue::Int32(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[MetadataValue]> {
        match self {
            MetadataValue::Array(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    pub fn display(&self) -> String {
        match self {
            MetadataValue::Uint8(v) => format!("{}", v),
            MetadataValue::Int8(v) => format!("{}", v),
            MetadataValue::Uint16(v) => format!("{}", v),
            MetadataValue::Int16(v) => format!("{}", v),
            MetadataValue::Uint32(v) => format!("{}", v),
            MetadataValue::Int32(v) => format!("{}", v),
            MetadataValue::Float32(v) => format!("{}", v),
            MetadataValue::Bool(v) => format!("{}", v),
            MetadataValue::String(v) => format!("\"{}\"", v),
            MetadataValue::Array(items) => {
                let inner: Vec<String> = items.iter().map(|x| x.display()).collect();
                format!("[{}]", inner.join(", "))
            }
            MetadataValue::Uint64(v) => format!("{}", v),
            MetadataValue::Int64(v) => format!("{}", v),
            MetadataValue::Float64(v) => format!("{}", v),
        }
    }
}

// ── Metadata KV pair (parsed) ─────────────────────────────────────────────

/// A single metadata key-value entry parsed from the GGUF header.
///
/// The value is decoded eagerly during `parse_metadata` so callers can inspect
/// it without maintaining a cursor into the mmap.
#[derive(Debug, Clone)]
pub struct KvEntry {
    pub key: String,
    pub value: MetadataValue,
}

// ── Tensor info ───────────────────────────────────────────────────────────

/// Descriptor for a single tensor in the GGUF tensor directory.
///
/// The tensor's payload bytes are not copied; they are accessed via
/// `abs_offset` into the mmap'd file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub ndim: u32,
    pub dims: Vec<u64>,
    pub tensor_type: u32,
    pub abs_offset: u64,
    pub rel_offset: u64,
    pub elements: u64,
    pub bytes: u64,
}

impl TensorInfo {
    /// Resolve the GGUF tensor format to the engine's `TensorType` enum, if
    /// the format is one that the inference kernels support.
    pub fn resolved_type(&self) -> Option<TensorType> {
        TensorType::from_u32(self.tensor_type)
    }

    /// Human-readable name for the underlying GGUF tensor type tag.
    pub fn type_name(&self) -> &'static str {
        TensorType::from_u32(self.tensor_type)
            .map(|t| t.name())
            .unwrap_or("unknown")
    }

    /// Return a slice pointing at the tensor's raw bytes in the mmap'd file.
    pub fn data<'a>(&self, mmap: &'a Mmap) -> &'a [u8] {
        let start = self.abs_offset as usize;
        let end = start + self.bytes as usize;
        &mmap[start..end]
    }
}

// ── GGUF model ────────────────────────────────────────────────────────────

/// A fully parsed GGUF v3 model loaded via mmap.
///
/// All tensor payload data remains in the mmap — the loader stores only
/// offsets and metadata.  Use `tensor_data(tensor)` or `TensorInfo::data()` to
/// obtain raw byte slices into the mapping.
pub struct ModelGGUF {
    /// Memory-mapped file contents.
    mmap: Mmap,
    /// GGUF format version (must be 3).
    pub version: u32,
    /// Number of metadata key-value entries.
    pub n_kv: u64,
    /// Number of tensor directory entries.
    pub n_tensors: u64,
    /// Tensor data alignment (default 32, may be overridden by metadata).
    pub alignment: u64,
    /// Byte offset within the mmap where tensor data begins.
    pub tensor_data_pos: u64,
    /// Parsed metadata key-value pairs.
    pub kv: Vec<KvEntry>,
    /// Parsed tensor directory entries.
    pub tensors: Vec<TensorInfo>,
}

impl ModelGGUF {
    /// Open a GGUF v3 model file and parse its header, metadata, and tensor
    /// directory.  The file is mmap'd and the mapping is held for the lifetime
    /// of `ModelGGUF`.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let file =
            File::open(path).with_context(|| format!("cannot open model: {}", path.display()))?;

        let file_len = file.metadata().context("cannot stat model")?.len();
        if file_len < 32 {
            bail!("model file is too small to be a valid GGUF file");
        }

        // Safety: the mmap is read-only and the file is not subsequently
        // truncated in our process.
        let mmap = unsafe { Mmap::map(&file) }.context("cannot mmap model")?;

        let mut model = ModelGGUF {
            mmap,
            version: 0,
            n_kv: 0,
            n_tensors: 0,
            alignment: 32,
            tensor_data_pos: 0,
            kv: Vec::new(),
            tensors: Vec::new(),
        };

        // ── Parse header ──────────────────────────────────────────────
        {
            let mut c = Cursor::new(&model.mmap, 0);

            let magic = c.read_u32().context("cannot read GGUF magic")?;
            if magic != DS4_GGUF_MAGIC {
                bail!("model is not a GGUF file (magic: {:#x})", magic);
            }

            model.version = c.read_u32().context("cannot read GGUF version")?;
            model.n_tensors = c.read_u64().context("cannot read n_tensors")?;
            model.n_kv = c.read_u64().context("cannot read n_kv")?;

            if model.version != 3 {
                bail!(
                    "only GGUF v3 is supported (found version {})",
                    model.version
                );
            }

            model.parse_metadata(&mut c)?;
            model.parse_tensors(&mut c)?;
        }

        Ok(model)
    }

    // ── Metadata parsing ──────────────────────────────────────────────

    /// Parse the metadata KV table.  During parsing we detect the special
    /// `general.alignment` key and update `self.alignment`.
    fn parse_metadata(&mut self, c: &mut Cursor) -> Result<()> {
        self.kv = Vec::with_capacity(self.n_kv as usize);

        for _ in 0..self.n_kv {
            // Read key (GGUF string: u64 length + bytes)
            let key = c.read_string().context("cannot read metadata key")?;

            // Read value type tag.
            let type_tag = c.read_u32().context("cannot read metadata value type")?;
            let value_type = MetadataType::from_u32(type_tag)
                .with_context(|| format!("unknown GGUF metadata type tag: {}", type_tag))?;

            // Record current position (start of value data) then skip over it
            // first so we can decode the value.
            let value = Self::read_metadata_value(c, &value_type, 0)
                .with_context(|| format!("cannot read metadata value for key '{}'", key))?;

            // Detect `general.alignment` override.
            if key == "general.alignment" {
                if let Some(align) = value.as_u32() {
                    if align != 0 {
                        self.alignment = u64::from(align);
                    }
                }
            }

            self.kv.push(KvEntry { key, value });
        }

        Ok(())
    }

    /// Recursively read a metadata value from the cursor.
    fn read_metadata_value(c: &mut Cursor, ty: &MetadataType, depth: u32) -> Result<MetadataValue> {
        if depth > 8 {
            bail!("metadata array nesting exceeds maximum depth of 8");
        }

        match ty {
            MetadataType::Uint8 => Ok(MetadataValue::Uint8(c.read_u8()?)),
            MetadataType::Int8 => Ok(MetadataValue::Int8(c.read_i8()?)),
            MetadataType::Uint16 => Ok(MetadataValue::Uint16(c.read_u16()?)),
            MetadataType::Int16 => Ok(MetadataValue::Int16(c.read_i16()?)),
            MetadataType::Uint32 => Ok(MetadataValue::Uint32(c.read_u32()?)),
            MetadataType::Int32 => Ok(MetadataValue::Int32(c.read_i32()?)),
            MetadataType::Float32 => Ok(MetadataValue::Float32(c.read_f32()?)),
            MetadataType::Bool => {
                let v = c.read_u8()?;
                Ok(MetadataValue::Bool(v != 0))
            }
            MetadataType::String => {
                let s = c.read_string()?;
                Ok(MetadataValue::String(s))
            }
            MetadataType::Array => {
                let item_tag = c.read_u32().context("cannot read array item type")?;
                let len = c.read_u64().context("cannot read array length")?;
                let item_type = MetadataType::from_u32(item_tag)
                    .with_context(|| format!("unknown GGUF array item type tag: {}", item_tag))?;

                // Pre-allocate and read items one by one. For scalar items
                // we could batch-read, but GGUF arrays are small so the
                // overhead is negligible and the code is simpler.
                let mut items = Vec::with_capacity(len as usize);
                for _ in 0..len {
                    let item = Self::read_metadata_value(c, &item_type, depth + 1)
                        .context("cannot read array item")?;
                    items.push(item);
                }

                Ok(MetadataValue::Array(items))
            }
            MetadataType::Uint64 => Ok(MetadataValue::Uint64(c.read_u64()?)),
            MetadataType::Int64 => Ok(MetadataValue::Int64(c.read_i64()?)),
            MetadataType::Float64 => Ok(MetadataValue::Float64(c.read_f64()?)),
        }
    }

    // ── Tensor directory parsing ──────────────────────────────────────

    /// Parse the tensor directory and convert relative GGUF offsets to absolute
    /// mmap offsets.
    fn parse_tensors(&mut self, c: &mut Cursor) -> Result<()> {
        self.tensors = Vec::with_capacity(self.n_tensors as usize);

        for _ in 0..self.n_tensors {
            let name = c.read_string().context("cannot read tensor name")?;

            let ndim = c.read_u32().context("cannot read tensor ndim")?;
            if ndim == 0 || ndim as usize > DS4_MAX_DIMS {
                bail!(
                    "tensor '{}' has unsupported number of dimensions: {}",
                    name,
                    ndim
                );
            }

            let mut dims = Vec::with_capacity(ndim as usize);
            let mut elements: u64 = 1;
            for _ in 0..ndim {
                let d = c.read_u64().context("cannot read tensor dimension")?;
                if d != 0 {
                    elements = elements
                        .checked_mul(d)
                        .context("tensor element count overflow")?;
                }
                dims.push(d);
            }

            let tensor_type = c.read_u32().context("cannot read tensor type")?;
            let rel_offset = c.read_u64().context("cannot read tensor offset")?;

            // Compute byte count from type info.
            let bytes = tensor_nbytes(tensor_type, elements).unwrap_or(0);

            self.tensors.push(TensorInfo {
                name,
                ndim,
                dims,
                tensor_type,
                abs_offset: 0, // filled after we know tensor_data_pos
                rel_offset,
                elements,
                bytes,
            });
        }

        // Compute tensor data region start (aligned).
        self.tensor_data_pos = align_up(c.pos as u64, self.alignment);

        // Convert relative offsets to absolute.
        for t in &mut self.tensors {
            t.abs_offset = self
                .tensor_data_pos
                .checked_add(t.rel_offset)
                .context("tensor offset overflow")?;

            if t.bytes != 0 {
                let end = t.abs_offset.checked_add(t.bytes).unwrap_or(u64::MAX);
                if end > self.mmap.len() as u64 {
                    bail!(
                        "tensor '{}' at offset {:#x} + {} bytes points beyond file (file length: {})",
                        t.name,
                        t.abs_offset,
                        t.bytes,
                        self.mmap.len()
                    );
                }
            }
        }

        Ok(())
    }

    // ── Lookup helpers ────────────────────────────────────────────────

    /// Find a tensor by name.  Returns `None` if no tensor with that name
    /// exists in the directory.
    pub fn find_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Find a metadata entry by key name.  Returns `None` if not found.
    pub fn find_kv(&self, key: &str) -> Option<&KvEntry> {
        self.kv.iter().find(|kv| kv.key == key)
    }

    /// Convenience: get a metadata value by key.
    pub fn kv_value(&self, key: &str) -> Option<&MetadataValue> {
        self.find_kv(key).map(|kv| &kv.value)
    }

    /// Convenience: read a string metadata value.
    pub fn get_string(&self, key: &str) -> Option<&str> {
        self.kv_value(key).and_then(|v| v.as_string())
    }

    /// Convenience: read a `u32` metadata value.
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        self.kv_value(key).and_then(|v| v.as_u32())
    }

    /// Convenience: read a `u64` metadata value.
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.kv_value(key).and_then(|v| v.as_u64())
    }

    /// Convenience: read a `bool` metadata value.
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.kv_value(key).and_then(|v| v.as_bool())
    }

    /// Convenience: read a `f32` metadata value.
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.kv_value(key).and_then(|v| v.as_f32())
    }

    /// Convenience: read an array metadata value.
    pub fn get_array(&self, key: &str) -> Option<&[MetadataValue]> {
        self.kv_value(key).and_then(|v| v.as_array())
    }

    /// Return a byte slice covering the raw tensor payload in the mmap.
    ///
    /// Panics if the tensor's byte range lies outside the mmap (should not
    /// happen because `parse_tensors` validates offsets).
    pub fn tensor_data<'a>(&'a self, tensor: &TensorInfo) -> &'a [u8] {
        let start = tensor.abs_offset as usize;
        let end = start + tensor.bytes as usize;
        &self.mmap[start..end]
    }

    /// Access the raw mmap.  Useful for callers that need to work with tensor
    /// offsets directly.
    pub fn mmap(&self) -> &Mmap {
        &self.mmap
    }

    // ── Summary ───────────────────────────────────────────────────────

    /// Print a human-readable summary of the model to stdout, mirroring
    /// `model_summary()` from `ds4.c`.
    pub fn summary(&self) {
        let name = self.get_string("general.name").unwrap_or("(unnamed)");
        let arch = self
            .get_string("general.architecture")
            .unwrap_or("(unknown)");
        let layers = self.get_u32("deepseek4.block_count").unwrap_or(0);
        let ctx_train = self.get_u64("deepseek4.context_length").unwrap_or(0);
        let n_head = self.get_u32("deepseek4.attention.head_count").unwrap_or(0);
        let n_head_kv = self
            .get_u32("deepseek4.attention.head_count_kv")
            .unwrap_or(0);
        let head_dim = self.get_u32("deepseek4.attention.key_length").unwrap_or(0);
        let n_swa = self
            .get_u32("deepseek4.attention.sliding_window")
            .unwrap_or(0);
        let indexer_heads = self
            .get_u32("deepseek4.attention.indexer.head_count")
            .unwrap_or(0);
        let indexer_head_dim = self
            .get_u32("deepseek4.attention.indexer.key_length")
            .unwrap_or(0);
        let indexer_top_k = self
            .get_u32("deepseek4.attention.indexer.top_k")
            .unwrap_or(0);
        let n_expert = self.get_u32("deepseek4.expert_count").unwrap_or(0);
        let n_expert_used = self.get_u32("deepseek4.expert_used_count").unwrap_or(0);
        let n_expert_groups = self.get_u32("deepseek4.expert_group_count").unwrap_or(0);
        let n_group_used = self
            .get_u32("deepseek4.expert_group_used_count")
            .unwrap_or(0);

        let mut tensor_bytes: u64 = 0;
        let mut params: u64 = 0;
        for t in &self.tensors {
            tensor_bytes += t.bytes;
            params += t.elements;
        }

        println!("model: {}", name);
        println!("arch:  {}", arch);
        println!(
            "gguf:  v{}, {} metadata keys, {} tensors",
            self.version, self.n_kv, self.n_tensors
        );
        if layers != 0 {
            println!("layers: {}", layers);
        }
        if ctx_train != 0 {
            println!("train context: {}", ctx_train);
        }
        if n_head != 0 || n_head_kv != 0 || head_dim != 0 || n_swa != 0 {
            println!(
                "attention: heads={} kv_heads={} head_dim={} swa={}",
                n_head, n_head_kv, head_dim, n_swa
            );
        }
        if indexer_heads != 0 || indexer_head_dim != 0 || indexer_top_k != 0 {
            println!(
                "indexer: heads={} head_dim={} top_k={}",
                indexer_heads, indexer_head_dim, indexer_top_k
            );
        }
        if n_expert != 0 || n_expert_used != 0 || n_expert_groups != 0 || n_group_used != 0 {
            println!(
                "experts: count={} used={} groups={} groups_used={}",
                n_expert, n_expert_used, n_expert_groups, n_group_used
            );
        }
        println!("file size: {}", format_bytes(self.mmap.len() as u64));
        println!(
            "tensor bytes described by GGUF: {}",
            format_bytes(tensor_bytes)
        );
        println!(
            "logical parameters: {:.2} B",
            params as f64 / 1_000_000_000.0
        );

        // Group tensors by type for display.
        use std::collections::BTreeMap;
        let mut by_type: BTreeMap<u32, (u64, u64)> = BTreeMap::new();
        for t in &self.tensors {
            let entry = by_type.entry(t.tensor_type).or_default();
            entry.0 += 1;
            entry.1 += t.bytes;
        }
        println!("tensor types:");
        for (type_tag, (count, bytes)) in &by_type {
            let type_name = TensorType::from_u32(*type_tag)
                .map(|t| t.name().to_string())
                .unwrap_or_else(|| format!("type_{}", type_tag));
            println!(
                "  {:<8} {:5} tensors, {}",
                type_name,
                count,
                format_bytes(*bytes)
            );
        }
    }

    /// Return the underlying mmap file length.
    pub fn file_size(&self) -> u64 {
        self.mmap.len() as u64
    }
}

// ── Helper: tensor byte count ─────────────────────────────────────────────

/// Compute the number of bytes a tensor occupies given its GGUF type tag and
/// element count.  Returns `None` for unsupported tensor type tags.
pub fn tensor_nbytes(type_tag: u32, elements: u64) -> Option<u64> {
    let block_elems = u64::from(type_block_elems(type_tag)?);
    let block_bytes = u64::from(type_block_bytes(type_tag)?);

    let blocks = (elements + block_elems - 1) / block_elems;
    blocks.checked_mul(block_bytes)
}

/// Return the number of elements per quantisation block for a GGUF tensor
/// type tag.  Mirrors `gguf_type_info.block_elems` from `ds4.c`.
fn type_block_elems(type_tag: u32) -> Option<u32> {
    match type_tag {
        0 => Some(1),    // F32
        1 => Some(1),    // F16
        2 => Some(32),   // Q4_0
        3 => Some(32),   // Q4_1
        6 => Some(32),   // Q5_0
        7 => Some(32),   // Q5_1
        8 => Some(32),   // Q8_0
        9 => Some(32),   // Q8_1
        10 => Some(256), // Q2_K
        11 => Some(256), // Q3_K
        12 => Some(256), // Q4_K
        13 => Some(256), // Q5_K
        14 => Some(256), // Q6_K
        15 => Some(256), // Q8_K
        16 => Some(256), // IQ2_XXS
        17 => Some(256), // IQ2_XS
        18 => Some(256), // IQ3_XXS
        19 => Some(256), // IQ1_S
        20 => Some(256), // IQ4_NL
        21 => Some(256), // IQ3_S
        22 => Some(256), // IQ2_S
        23 => Some(256), // IQ4_XS
        24 => Some(1),   // I8
        25 => Some(1),   // I16
        26 => Some(1),   // I32
        27 => Some(1),   // I64
        28 => Some(1),   // F64
        29 => Some(256), // IQ1_M
        30 => Some(1),   // BF16
        _ => None,
    }
}

/// Return the number of bytes per quantisation block for a GGUF tensor type
/// tag.  Mirrors `gguf_type_info.block_bytes` from `ds4.c`.
fn type_block_bytes(type_tag: u32) -> Option<u32> {
    match type_tag {
        0 => Some(4),    // F32
        1 => Some(2),    // F16
        2 => Some(18),   // Q4_0
        3 => Some(20),   // Q4_1
        6 => Some(22),   // Q5_0
        7 => Some(24),   // Q5_1
        8 => Some(34),   // Q8_0
        9 => Some(40),   // Q8_1
        10 => Some(84),  // Q2_K
        11 => Some(110), // Q3_K
        12 => Some(144), // Q4_K
        13 => Some(176), // Q5_K
        14 => Some(210), // Q6_K
        15 => Some(292), // Q8_K
        16 => Some(66),  // IQ2_XXS
        17 => Some(74),  // IQ2_XS
        18 => Some(98),  // IQ3_XXS
        19 => Some(110), // IQ1_S
        20 => Some(50),  // IQ4_NL
        21 => Some(110), // IQ3_S
        22 => Some(82),  // IQ2_S
        23 => Some(136), // IQ4_XS
        24 => Some(1),   // I8
        25 => Some(2),   // I16
        26 => Some(4),   // I32
        27 => Some(8),   // I64
        28 => Some(8),   // F64
        29 => Some(56),  // IQ1_M
        30 => Some(2),   // BF16
        _ => None,
    }
}

// ── Alignment helper ──────────────────────────────────────────────────────

/// Round `value` up to the next multiple of `alignment`.
pub fn align_up(value: u64, alignment: u64) -> u64 {
    let rem = value % alignment;
    if rem == 0 {
        value
    } else {
        value + alignment - rem
    }
}

// ── Byte-size formatting ──────────────────────────────────────────────────

/// Format a byte count as a human-readable GiB string, matching
/// `print_size()` from `ds4.c`.
pub fn format_bytes(bytes: u64) -> String {
    let gib = 1024.0 * 1024.0 * 1024.0;
    format!("{:.2} GiB", bytes as f64 / gib)
}

// ── Internal cursor ───────────────────────────────────────────────────────

/// A small cursor over a byte slice used to parse GGUF header/metadata fields
/// sequentially, mirroring `ds4_cursor` from `ds4.c`.
struct Cursor {
    data: *const u8,
    len: usize,
    pos: usize,

    // Hold a reference to keep the mmap alive. We maintain this as a raw
    // pointer for convenience but tie the lifetime to `ModelGGUF`.
    _mmap: std::marker::PhantomData<Mmap>,
}

// SAFETY: Cursor only reads from the mmap-backed data and never writes.
unsafe impl Send for Cursor {}
unsafe impl Sync for Cursor {}

impl Cursor {
    /// Create a new cursor at `offset` into the mmap.
    fn new(mmap: &Mmap, offset: usize) -> Self {
        Cursor {
            data: mmap.as_ptr(),
            len: mmap.len(),
            pos: offset,
            _mmap: std::marker::PhantomData,
        }
    }

    /// Remaining bytes from the current position.
    fn remaining(&self) -> usize {
        self.len.saturating_sub(self.pos)
    }

    /// Check that `n` bytes can be read, and advance `pos`.
    fn advance(&mut self, n: usize) -> Result<()> {
        if n > self.remaining() {
            bail!(
                "truncated GGUF file: needed {} bytes, {} remaining at pos {}",
                n,
                self.remaining(),
                self.pos
            );
        }
        self.pos += n;
        Ok(())
    }

    /// Return a slice view starting at the current position (peek without
    /// advancing).
    fn view(&self, n: usize) -> Result<&[u8]> {
        if n > self.remaining() {
            bail!(
                "truncated GGUF file: needed {} bytes, {} remaining at pos {}",
                n,
                self.remaining(),
                self.pos
            );
        }
        // Safety: the mmap lives longer than the cursor.
        Ok(unsafe { std::slice::from_raw_parts(self.data.add(self.pos), n) })
    }

    /// Read a `u8` in native byte order (single byte).
    fn read_u8(&mut self) -> Result<u8> {
        let slice = self.view(1)?;
        let v = slice[0];
        self.pos += 1;
        Ok(v)
    }

    /// Read an `i8`.
    fn read_i8(&mut self) -> Result<i8> {
        let slice = self.view(1)?;
        let v = slice[0] as i8;
        self.pos += 1;
        Ok(v)
    }

    /// Read a little-endian `u16`.
    fn read_u16(&mut self) -> Result<u16> {
        let slice = self.view(2)?;
        let v = (&slice[..]).read_u16::<LittleEndian>()?;
        self.pos += 2;
        Ok(v)
    }

    /// Read a little-endian `i16`.
    fn read_i16(&mut self) -> Result<i16> {
        let slice = self.view(2)?;
        let v = (&slice[..]).read_i16::<LittleEndian>()?;
        self.pos += 2;
        Ok(v)
    }

    /// Read a little-endian `u32`.
    fn read_u32(&mut self) -> Result<u32> {
        let slice = self.view(4)?;
        let v = (&slice[..]).read_u32::<LittleEndian>()?;
        self.pos += 4;
        Ok(v)
    }

    /// Read a little-endian `i32`.
    fn read_i32(&mut self) -> Result<i32> {
        let slice = self.view(4)?;
        let v = (&slice[..]).read_i32::<LittleEndian>()?;
        self.pos += 4;
        Ok(v)
    }

    /// Read a little-endian `f32`.
    fn read_f32(&mut self) -> Result<f32> {
        let slice = self.view(4)?;
        let v = (&slice[..]).read_f32::<LittleEndian>()?;
        self.pos += 4;
        Ok(v)
    }

    /// Read a little-endian `u64`.
    fn read_u64(&mut self) -> Result<u64> {
        let slice = self.view(8)?;
        let v = (&slice[..]).read_u64::<LittleEndian>()?;
        self.pos += 8;
        Ok(v)
    }

    /// Read a little-endian `i64`.
    fn read_i64(&mut self) -> Result<i64> {
        let slice = self.view(8)?;
        let v = (&slice[..]).read_i64::<LittleEndian>()?;
        self.pos += 8;
        Ok(v)
    }

    /// Read a little-endian `f64`.
    fn read_f64(&mut self) -> Result<f64> {
        let slice = self.view(8)?;
        let v = (&slice[..]).read_f64::<LittleEndian>()?;
        self.pos += 8;
        Ok(v)
    }

    /// Read a GGUF string: a `u64` length prefix followed by that many UTF-8
    /// bytes.
    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()? as usize;
        let slice = self.view(len)?;
        let s = String::from_utf8(slice.to_vec())
            .with_context(|| format!("metadata string at pos {} is not valid UTF-8", self.pos))?;
        self.pos += len;
        Ok(s)
    }

    /// Skip `n` bytes.
    fn skip(&mut self, n: u64) -> Result<()> {
        self.advance(n as usize)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 32), 0);
        assert_eq!(align_up(1, 32), 32);
        assert_eq!(align_up(32, 32), 32);
        assert_eq!(align_up(33, 32), 64);
        assert_eq!(align_up(0, 256), 0);
        assert_eq!(align_up(255, 256), 256);
        assert_eq!(align_up(256, 256), 256);
    }

    #[test]
    fn test_format_bytes() {
        let s = format_bytes(1024 * 1024 * 1024);
        assert_eq!(s, "1.00 GiB");
    }

    #[test]
    fn test_metadata_type_roundtrip() {
        for (tag, expected) in [
            (0, MetadataType::Uint8),
            (1, MetadataType::Int8),
            (4, MetadataType::Uint32),
            (6, MetadataType::Float32),
            (7, MetadataType::Bool),
            (8, MetadataType::String),
            (9, MetadataType::Array),
            (12, MetadataType::Float64),
        ] {
            assert_eq!(MetadataType::from_u32(tag), Some(expected));
            assert_eq!(expected as u32, tag);
        }
        assert_eq!(MetadataType::from_u32(99), None);
    }

    #[test]
    fn test_scalar_size() {
        assert_eq!(MetadataType::Uint8.scalar_size(), Some(1));
        assert_eq!(MetadataType::Int8.scalar_size(), Some(1));
        assert_eq!(MetadataType::Bool.scalar_size(), Some(1));
        assert_eq!(MetadataType::Uint16.scalar_size(), Some(2));
        assert_eq!(MetadataType::Int16.scalar_size(), Some(2));
        assert_eq!(MetadataType::Uint32.scalar_size(), Some(4));
        assert_eq!(MetadataType::Int32.scalar_size(), Some(4));
        assert_eq!(MetadataType::Float32.scalar_size(), Some(4));
        assert_eq!(MetadataType::Uint64.scalar_size(), Some(8));
        assert_eq!(MetadataType::Int64.scalar_size(), Some(8));
        assert_eq!(MetadataType::Float64.scalar_size(), Some(8));
        assert_eq!(MetadataType::String.scalar_size(), None);
        assert_eq!(MetadataType::Array.scalar_size(), None);
    }

    #[test]
    fn test_metadata_value_conversions() {
        let v = MetadataValue::Uint32(42);
        assert_eq!(v.as_u32(), Some(42));
        assert_eq!(v.as_f32(), Some(42.0));
        assert!(v.as_bool().is_none());
        assert!(v.as_string().is_none());

        let v = MetadataValue::Float32(3.14);
        assert_eq!(v.as_f32(), Some(3.14));
        assert!(v.as_u32().is_none());

        let v = MetadataValue::Bool(true);
        assert_eq!(v.as_bool(), Some(true));

        let v = MetadataValue::String("hello".to_string());
        assert_eq!(v.as_string(), Some("hello"));

        let v = MetadataValue::Uint64(999);
        assert_eq!(v.as_u64(), Some(999));
    }

    #[test]
    fn test_tensor_nbytes() {
        // F32: 1 element = 4 bytes
        assert_eq!(tensor_nbytes(0, 1), Some(4));
        // F32: 100 elements = 400 bytes
        assert_eq!(tensor_nbytes(0, 100), Some(400));
        // F16: 1 element = 2 bytes
        assert_eq!(tensor_nbytes(1, 1), Some(2));
        // Q8_0: 32 elements = 34 bytes
        assert_eq!(tensor_nbytes(8, 32), Some(34));
        // Q8_0: 33 elements = 2 blocks = 68 bytes
        assert_eq!(tensor_nbytes(8, 33), Some(68));
        // Q2_K: 256 elements = 84 bytes
        assert_eq!(tensor_nbytes(10, 256), Some(84));
        // IQ2_XXS: 256 elements = 66 bytes
        assert_eq!(tensor_nbytes(16, 256), Some(66));
        // Unknown type returns None
        assert!(tensor_nbytes(255, 10).is_none());
    }

    #[test]
    fn test_metadata_value_display() {
        assert_eq!(MetadataValue::Uint32(42).display(), "42");
        assert_eq!(MetadataValue::Bool(true).display(), "true");
        assert_eq!(MetadataValue::String("test".into()).display(), "\"test\"");
        assert_eq!(
            MetadataValue::Array(vec![MetadataValue::Uint32(1), MetadataValue::Uint32(2),])
                .display(),
            "[1, 2]"
        );
    }
}
