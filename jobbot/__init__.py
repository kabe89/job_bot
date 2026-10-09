"""JobBot — automated job search & resume tailoring."""
__version__ = "1.0.0"

# Ensure Python SSL verification uses the Windows native Certificate Store
# to prevent SSLCertVerificationError on LinkedIn and external job boards.
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

from .apply_runner import launch_copilot, run_apply
