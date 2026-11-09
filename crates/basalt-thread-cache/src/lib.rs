//! Cache identities are derived from *actual* token IDs and immutable engine
//! configuration fingerprints. Thread IDs and human model names are never safe
//! cache-equivalence keys. Caller must supply a VERIFIED model-state digest.
use sha2::{Digest, Sha256};
use uuid::Uuid;

pub const IDENTITY_VERSION: u32 = 1;

#[derive(Debug, Clone)]
pub struct CacheIdentityInput<'a> {
    pub owner: Uuid,
    pub model_state_sha256: [u8;32],
    pub template_sha256: [u8;32],
    pub kv_config_sha256: [u8;32],
    pub token_ids: &'a [u32],
    pub multimodal_content_sha256: Option<[u8;32]>,
}

/// Length-delimited encoding and a version/domain label prevent concatenation
/// ambiguity and allow future incompatible hash formats to coexist.
pub fn identity(data: &CacheIdentityInput<'_>) -> [u8;32] {
    let mut h=Sha256::new();
    h.update(b"basalt.thread.cache.identity");
    h.update(IDENTITY_VERSION.to_be_bytes());
    h.update(data.owner.as_bytes());
    h.update(data.model_state_sha256);
    h.update(data.template_sha256);
    h.update(data.kv_config_sha256);
    h.update((data.token_ids.len() as u64).to_be_bytes());
    for token in data.token_ids { h.update(token.to_be_bytes()); }
    match data.multimodal_content_sha256 {
        None => h.update([0u8]),
        Some(digest) => { h.update([1u8]); h.update(digest); }
    }
    h.finalize().into()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Residency {
    Gpu { slot: u32 },
    Host { allocation: Uuid },
    Disk,
    Missing,
}
#[derive(Debug, Clone)]
pub struct CacheManifest {
    pub identity: [u8;32],
    pub engine_epoch: Uuid,
    pub format_version: u32,
    pub prefix_tokens: u64,
    pub residency: Residency,
    pub validated: bool,
}
impl CacheManifest {
    /// In-process GPU/RAM state from an earlier process epoch is never resumed.
    /// A disk checkpoint needs an independent format and checksum validation.
    pub fn usable(&self, active_epoch: Uuid, format_version: u32) -> bool {
        self.validated && self.format_version==format_version
            && match self.residency {
                Residency::Gpu{..} | Residency::Host{..} => self.engine_epoch==active_epoch,
                Residency::Disk => false,  // engine-side proof/restore is required
                Residency::Missing => false,
            }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn base<'a>(ids:&'a [u32]) -> CacheIdentityInput<'a> {
        CacheIdentityInput {
            owner:Uuid::from_u128(1), model_state_sha256:[2;32],
            template_sha256:[3;32], kv_config_sha256:[4;32],
            token_ids:ids, multimodal_content_sha256:None,
        }
    }
    #[test]
    fn all_identity_dimensions_change_key() {
        let ids=[10,20,30];let original=base(&ids);
        let key=identity(&original);
        let mut changed=original.clone();
        changed.owner=Uuid::from_u128(2);assert_ne!(identity(&changed),key);
        changed=original.clone(); changed.model_state_sha256=[9;32];assert_ne!(identity(&changed),key);
        changed=original.clone(); changed.template_sha256=[9;32];assert_ne!(identity(&changed),key);
        changed=original.clone(); changed.kv_config_sha256=[9;32];assert_ne!(identity(&changed),key);
        changed=original.clone(); changed.multimodal_content_sha256=Some([9;32]);assert_ne!(identity(&changed),key);
        assert_ne!(identity(&base(&[10,20,31])),key);
        assert_ne!(identity(&base(&[10,20])),key);
    }
    #[test]
    fn invalid_or_stale_manifest_cannot_restore() {
        let epoch=Uuid::new_v4();
        let mut m=CacheManifest {identity:[5;32],engine_epoch:epoch,
          format_version:1,prefix_tokens:1024,residency:Residency::Gpu{slot:0},validated:false};
        assert!(!m.usable(epoch,1));
        m.validated=true; assert!(m.usable(epoch,1));
        assert!(!m.usable(Uuid::new_v4(),1));
        assert!(!m.usable(epoch,2));
        m.residency=Residency::Disk;assert!(!m.usable(epoch,1));
    }
}
