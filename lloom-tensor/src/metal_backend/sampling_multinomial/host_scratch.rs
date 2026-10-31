//! Sampler-local immutable-capacity host storage; no public demand/allocator API.
use crate::cold_bytes::{ColdBytesError, ColdScratch, ScratchScalar};
use crate::metal_backend::{allocation_scope, MetalDevice};
use crate::Result;

pub(super) enum HostScratch<T: ScratchScalar + Default> {
    Supplied(ColdScratch<T>),
    Ordinary(Vec<T>),
}
impl<T: ScratchScalar + Default> HostScratch<T> {
    pub(super) fn new(device: &MetalDevice, count: usize) -> Result<Self> {
        if let Some(scope) = allocation_scope::active(device)? {
            let values =
                ColdScratch::zeroed_in(&scope.context, count).map_err(|error| match error {
                    ColdBytesError::Admission(error) => crate::Error::from(error),
                    error => {
                        crate::Error::Msg(format!("sampling host scratch allocation: {error:?}"))
                    }
                })?;
            return Ok(Self::Supplied(values));
        }
        // Preserve ordinary callers: absence of supplied context is not license
        // to invent a second promise or claim legacy machine admission.
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|e| crate::Error::Msg(format!("sampling host scratch capacity: {e}")))?;
        values.resize(count, T::default());
        Ok(Self::Ordinary(values))
    }
}
impl<T: ScratchScalar + Default> std::ops::Deref for HostScratch<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        match self {
            Self::Supplied(values) => values,
            Self::Ordinary(values) => values,
        }
    }
}
impl<T: ScratchScalar + Default> std::ops::DerefMut for HostScratch<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        match self {
            Self::Supplied(values) => values,
            Self::Ordinary(values) => values,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_admission::{AllocationError, ModelDomain, RefusalDomain};
    use crate::metal_backend::MetalAllocationDomain;

    fn device() -> Result<crate::Device> {
        crate::isolation::claim("sampler host scratch ownership");
        let mut tuning = crate::tuning::Tuning::default();
        tuning.from_env().expect("valid fixture tuning");
        tuning.contract.profile = "deterministic".into();
        crate::tuning::install(tuning).expect("before device");
        let d = crate::Device::new_metal(0)?;
        println!(
            "HOST_SCRATCH_SETUP actual_api={}",
            d.as_metal_device()?.commands().actual_command_api()
        );
        Ok(d)
    }

    #[test]
    fn ordinary_scratch_keeps_initialized_fixed_scalar_storage() -> Result<()> {
        let d = device()?;
        let mut ints = HostScratch::<u32>::new(d.as_metal_device()?, 7)?;
        let mut floats = HostScratch::<f32>::new(d.as_metal_device()?, 7)?;
        assert!(matches!(&ints, HostScratch::Ordinary(_)));
        assert!(matches!(&floats, HostScratch::Ordinary(_)));
        assert_eq!(&*ints, &[0; 7]);
        assert!(floats.iter().all(|x| x.to_bits() == 0));
        ints[6] = u32::MAX;
        floats[6] = -3.5;
        assert_eq!(ints[6], u32::MAX);
        assert_eq!(floats[6].to_bits(), (-3.5f32).to_bits());
        Ok(())
    }

    #[test]
    fn supplied_scratch_partitions_exact_promise_and_keeps_live_charge() -> Result<()> {
        let d = device()?;
        let namespace = MetalAllocationDomain::new(
            d.as_metal_device()?,
            ModelDomain::isolated_numeric_fixture(64 << 20),
        )?;
        let baseline = namespace.domain().snapshot();
        let quote = ColdScratch::<u32>::requested_bytes(7).expect("audited scalar/control layout");
        assert_eq!(quote, ColdScratch::<f32>::requested_bytes(7).unwrap());
        let short = namespace.domain().reserve(quote - 1)?;
        let scope = namespace.scope(short.context())?;
        scope.run(|m| {
            let error = match HostScratch::<u32>::new(m, 7) {
                Err(error) => error,
                Ok(_) => panic!("HOST_SCRATCH_UNPAID_BIRTH"),
            };
            assert!(
                matches!(error, crate::Error::Allocation(AllocationError::Refused {
                requested, available, domain: RefusalDomain::Promise,
            }) if requested == quote && available == quote - 1),
                "{error}"
            );
            Ok(())
        })?;
        drop(short);
        drop(scope);
        assert_eq!(namespace.domain().snapshot(), baseline);
        let pass = namespace.domain().reserve(2 * quote)?;
        let scope = namespace.scope(pass.context())?;
        let (mut ints, mut floats) = scope.run(|m| {
            Ok((
                HostScratch::<u32>::new(m, 7)?,
                HostScratch::<f32>::new(m, 7)?,
            ))
        })?;
        assert!(matches!(&ints, HostScratch::Supplied(_)));
        assert!(matches!(&floats, HostScratch::Supplied(_)));
        assert_eq!(&*ints, &[0; 7]);
        assert!(floats.iter().all(|x| x.to_bits() == 0));
        ints[0] = 17;
        floats[0] = -0.;
        assert_eq!(ints[0], 17);
        assert_eq!(floats[0].to_bits(), (-0f32).to_bits());
        drop(pass);
        let live = namespace.domain().snapshot();
        assert_eq!(live.pending, baseline.pending);
        assert_eq!(live.committed, baseline.committed + 2 * quote);
        let error = scope
            .run(|m| HostScratch::<u32>::new(m, 7))
            .err()
            .expect("closed scope refuses birth");
        assert!(matches!(
            error,
            crate::Error::Allocation(AllocationError::Cancelled)
        ));
        drop(ints);
        drop(floats);
        drop(scope);
        assert_eq!(namespace.domain().snapshot(), baseline);
        Ok(())
    }

    #[test]
    fn supplied_scratch_cannot_fallback_for_foreign_device_or_missing_scope() -> Result<()> {
        let d = device()?;
        let other = crate::Device::new_metal(0)?;
        let namespace = MetalAllocationDomain::new(
            d.as_metal_device()?,
            ModelDomain::isolated_numeric_fixture(64 << 20),
        )?;
        let quote = ColdScratch::<u32>::requested_bytes(7).unwrap();
        let pass = namespace.domain().reserve(quote)?;
        let scope = namespace.scope(pass.context())?;
        let before = namespace.domain().snapshot();
        let bound = scope.run(|m| {
            let error = HostScratch::<u32>::new(other.as_metal_device()?, 7)
                .err()
                .expect("foreign device refuses");
            assert!(
                error.to_string().contains("inside execution scope"),
                "{error}"
            );
            Ok(m.clone())
        })?;
        let error = HostScratch::<u32>::new(&bound, 7)
            .err()
            .expect("missing scope refuses");
        assert!(
            error
                .to_string()
                .contains("requires an active execution scope"),
            "{error}"
        );
        assert_eq!(namespace.domain().snapshot(), before);
        Ok(())
    }
}
