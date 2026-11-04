//! External-consumer controls for the real public scoped residency surface.
//! CPU tensors/actual routed kernels, not the standalone interface prototype.
use lloom::nn::expert_stream::{
    ErasedResolvedExperts, ErasedResolvedProjection, ExpertResidency, Resident, ResolvedExperts,
    RoutedGroup, SendResolvedExperts,
};
use lloom_tensor::quantized::{GgmlDType, MoeArm, QTensor};
use lloom_tensor::{DType, Device, Result, Tensor};
use std::cell::{Cell, RefCell};
use std::num::NonZeroU64;
use std::sync::Arc;

/// Deliberately not Sync. It owns actual immutable resident weights and one
/// local checked lease, proving the public trait is not silently Sync-sealed.
struct LocalProvider {
    resident: Resident,
    gate: Arc<QTensor>,
    up: Arc<QTensor>,
    down: Arc<QTensor>,
    source_ids: RefCell<Option<Tensor>>,
    active: Cell<Option<NonZeroU64>>,
    next: Cell<u64>,
    aborts: Cell<usize>,
    successes: Cell<usize>,
}
impl LocalProvider {
    fn new() -> Self {
        let make = || {
            Arc::new(
                QTensor::quantize(
                    &Tensor::ones((2, 32, 32), DType::F32, &Device::Cpu).unwrap(),
                    GgmlDType::F32,
                )
                .unwrap(),
            )
        };
        let (gate, up, down) = (make(), make(), make());
        Self {
            resident: Resident::new(gate.clone(), up.clone(), down.clone()),
            gate,
            up,
            down,
            source_ids: RefCell::new(None),
            active: Cell::new(None),
            next: Cell::new(0),
            aborts: Cell::new(0),
            successes: Cell::new(0),
        }
    }
    fn authority(&self) -> usize {
        std::ptr::from_ref(self).addr()
    }
    fn validate(&self, lease: (usize, Option<NonZeroU64>)) -> Result<()> {
        if lease.0 != self.authority() || lease.1.is_none() || self.active.get() != lease.1 {
            lloom_tensor::bail!("external local group has a different/closed lease")
        }
        Ok(())
    }
    fn complete(&self, lease: (usize, Option<NonZeroU64>), success: bool) -> Result<()> {
        self.validate(lease)?;
        self.active.set(None);
        if success {
            self.successes.set(self.successes.get() + 1);
        } else {
            self.aborts.set(self.aborts.get() + 1);
        }
        Ok(())
    }
    fn validate_erased(&self, group: &ErasedResolvedProjection<'_, '_>) -> Result<()> {
        if group.owner_identity() != self.resolved_owner_identity() {
            lloom_tensor::bail!("external erased group has a different owner")
        }
        self.validate(group.host_lease())
    }
}
impl ExpertResidency for LocalProvider {
    fn abort_resolved_host_group(&self, authority: usize, generation: NonZeroU64) {
        if authority == self.authority() && self.active.get() == Some(generation) {
            self.active.set(None);
            self.aborts.set(self.aborts.get() + 1);
        }
    }
    fn plan(&self, ids: &Tensor) -> Result<Vec<RoutedGroup>> {
        let groups = self.resident.plan(ids)?;
        // This provider retains its OWN immutable router input, not IDs
        // detached from another provider's resolved custody.
        self.source_ids.replace(Some(ids.clone()));
        Ok(groups)
    }
    fn ensure(&self, group: &RoutedGroup) -> Result<ResolvedExperts<'_, Self>> {
        if self.active.get().is_some() {
            lloom_tensor::bail!("external local group still active")
        }
        let source = self.source_ids.borrow();
        let source = source.as_ref().ok_or_else(|| {
            lloom_tensor::Error::Msg("external local provider has no planned input".into())
        })?;
        let ids = source.narrow(0, group.rows.start, group.rows.len())?;
        let next = self
            .next
            .get()
            .checked_add(1)
            .and_then(NonZeroU64::new)
            .ok_or_else(|| lloom_tensor::Error::Msg("external generation exhausted".into()))?;
        self.next.set(next.get());
        self.active.set(Some(next));
        // SAFETY: exact borrowed owner + nonwrapping active generation; abort
        // checks the same lease, and our weights outlive every returned guard.
        Ok(unsafe {
            ResolvedExperts::from_provider_unchecked(self, ids, self.authority(), Some(next))
        })
    }
    fn ensure_erased(&self, group: &RoutedGroup) -> Result<ErasedResolvedExperts<'_>> {
        self.ensure(group).map(|group| group.into_erased())
    }
    fn gate_up(
        &self,
        x: &Tensor,
        group: &mut ResolvedExperts<'_, Self>,
        arm: MoeArm,
        act: DType,
    ) -> Result<(Tensor, Tensor)> {
        self.validate(group.host_lease())?;
        // SAFETY: validated exact active lease; no IDs escape this dispatch.
        let ids = unsafe { group.ids_for_provider_unchecked() };
        self.gate
            .indexed_moe_forward_pair_as(&self.up, x, ids, arm, act)
    }
    fn gate_up_erased(
        &self,
        x: &Tensor,
        group: &mut ErasedResolvedProjection<'_, '_>,
        arm: MoeArm,
        act: DType,
    ) -> Result<(Tensor, Tensor)> {
        self.validate_erased(group)?;
        // SAFETY: same actual owner/lease and immutable owned resident weights.
        let ids = unsafe { group.ids_for_provider_unchecked() };
        self.gate
            .indexed_moe_forward_pair_as(&self.up, x, ids, arm, act)
    }
    fn down(
        &self,
        h: &Tensor,
        group: &mut ResolvedExperts<'_, Self>,
        arm: MoeArm,
    ) -> Result<Tensor> {
        let lease = group.host_lease();
        self.validate(lease)?;
        let out = self.down.indexed_moe_forward_with(
            h,
            // SAFETY: exact validated lease; CPU dispatch finishes synchronously.
            unsafe { group.ids_for_provider_unchecked() },
            arm,
        );
        self.complete(lease, out.is_ok())?;
        // SAFETY: exact CPU generation is now closed, including the Err path.
        unsafe { group.close_for_provider_unchecked() };
        out
    }
    fn down_erased(
        &self,
        h: &Tensor,
        group: &mut ErasedResolvedProjection<'_, '_>,
        arm: MoeArm,
    ) -> Result<Tensor> {
        self.validate_erased(group)?;
        let lease = group.host_lease();
        let out = self.down.indexed_moe_forward_with(
            h,
            // SAFETY: exact validated owner/lease; IDs do not escape.
            unsafe { group.ids_for_provider_unchecked() },
            arm,
        );
        self.complete(lease, out.is_ok())?;
        // SAFETY: shared exact CPU close mechanism ran above.
        unsafe { group.close_for_provider_unchecked() };
        out
    }
}
fn flat(tensor: Tensor) -> Vec<f32> {
    tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

#[test]
fn real_public_non_sync_erasure_keeps_payload_and_exactly_once_cleanup() {
    let provider = LocalProvider::new();
    let object: &dyn ExpertResidency = &provider;
    let ids = Tensor::new(&[[0u32]], &Device::Cpu).unwrap();
    let x = Tensor::ones((1, 1, 32), DType::F32, &Device::Cpu).unwrap();
    let planned = object.plan(&ids).unwrap();
    let mut typed = provider.ensure(&planned[0]).unwrap();
    let (g, _) = provider
        .gate_up(&x, &mut typed, MoeArm::MatVec, DType::F32)
        .unwrap();
    let expected = flat(provider.down(&g, &mut typed, MoeArm::MatVec).unwrap());
    drop(typed);
    assert_eq!((provider.successes.get(), provider.aborts.get()), (1, 0));

    let local = object.ensure_erased(&planned[0]).unwrap();
    let local = match local.try_into_send() {
        Err(original) => original,
        Ok(_) => panic!("non-Sync original cleanup authority became Send"),
    };
    assert!(
        provider.active.get().is_some(),
        "failed extraction lost custody"
    );
    drop(local);
    assert_eq!(
        provider.aborts.get(),
        1,
        "failed extraction aborts exactly once"
    );
    assert!(provider.active.get().is_none());
    let mut local = object.ensure_erased(&planned[0]).unwrap();
    let (g, _) = object
        .gate_up_erased(&x, &mut local.as_projection(), MoeArm::MatVec, DType::F32)
        .unwrap();
    let got = object
        .down_erased(&g, &mut local.as_projection(), MoeArm::MatVec)
        .unwrap();
    assert_eq!(flat(got), expected);
    drop(local);
    assert_eq!((provider.successes.get(), provider.aborts.get()), (2, 1));
}

#[test]
fn real_public_auto_traits_preserve_typed_send_and_local_non_send() {
    fn send<T: Send>() {}
    send::<ResolvedExperts<'_, Resident>>();
    send::<SendResolvedExperts<'_>>();
    // The ambiguity contracts compile iff these real public types are NOT Send.
    trait AmbiguousIfSend<A> {
        fn check() {}
    }
    impl<T: ?Sized> AmbiguousIfSend<()> for T {}
    impl<T: ?Sized + Send> AmbiguousIfSend<u8> for T {}
    let _ = <ResolvedExperts<'_, LocalProvider> as AmbiguousIfSend<_>>::check;
    let _ = <ErasedResolvedExperts<'_> as AmbiguousIfSend<_>>::check;
}
