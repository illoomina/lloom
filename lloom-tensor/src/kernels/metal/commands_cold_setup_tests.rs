//! Device-free checked-resolver controls. Constructor ordering is source-audited
//! separately: these tests do not claim to execute a native constructor.
use super::MetalKernelError;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

const HELPER: &str = "kernels::metal::commands::cold_setup_tests::malformed_environment_helper";

// Same owned-child cleanup pattern as machine_process_tests; this child only
// resolves host configuration and can never hold a GPU wait or native owner.
struct Helper(Option<Child>);
impl Drop for Helper {
    fn drop(&mut self) {
        if let Some(mut child) = self.0.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

#[test]
fn malformed_environment_is_typed_without_native_construction_or_install() {
    let mut command = Command::new(std::env::current_exe().unwrap());
    command.args([
        "--exact",
        HELPER,
        "--ignored",
        "--nocapture",
        "--test-threads=1",
    ]);
    // Only the child environment changes. Keep loader/Cargo paths but remove
    // inherited tuning aliases/canonical values, so the one malformed knob
    // below is the tested cause rather than another process's configuration.
    for alias in crate::options::env_aliases(crate::tuning::Tuning::META) {
        command.env_remove(alias);
    }
    for (name, _) in std::env::vars_os() {
        let text = name.to_string_lossy();
        if text.starts_with("LLOOM_") || text.starts_with("CANDLE_") {
            command.env_remove(name);
        }
    }
    command.env("LLOOM_CB_UNRETAINED", "not-a-bool");
    let mut helper = Helper(Some(command.spawn().unwrap()));
    let deadline = Instant::now() + Duration::from_secs(5);
    let status = loop {
        if let Some(status) = helper.0.as_mut().unwrap().try_wait().unwrap() {
            break status;
        }
        assert!(
            Instant::now() < deadline,
            "cold resolver helper exit deadline"
        );
        std::thread::sleep(Duration::from_millis(5));
    };
    helper.0.take().unwrap().wait().unwrap();
    assert!(status.success(), "cold resolver helper: {status}");
}

#[test]
#[ignore = "isolated configuration helper; invoked by bounded parent test"]
fn malformed_environment_helper() {
    assert_eq!(std::env::var("LLOOM_CB_UNRETAINED").unwrap(), "not-a-bool");
    for attempt in 1..=2 {
        assert!(!crate::tuning::installed());
        let error = crate::tuning::try_tuning().unwrap_err();
        assert!(matches!(
            &error,
            crate::tuning::TuningSetupError::InvalidEnvironment(reason)
                if reason.contains("LLOOM_CB_UNRETAINED")
        ));
        let mapped = MetalKernelError::InvalidInput(error.to_string());
        assert!(matches!(
            &mapped,
            MetalKernelError::InvalidInput(reason) if reason.contains("LLOOM_CB_UNRETAINED")
        ));
        assert!(!crate::tuning::installed());
        eprintln!(
            "COLD_SETUP_HELPER attempt={attempt} variant=InvalidEnvironment mapped=InvalidInput installed=false profile_compiled={} cause={mapped}",
            crate::tuning::PROFILE_INSTRUMENTS
        );
    }
}
