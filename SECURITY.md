# Security Policy

## Supported Versions

Security fixes are provided for currently maintained releases.

| Version | Supported |
| --- | --- |
| 0.1.x | Yes |
| < 0.1.0 | No |
| `main` (unreleased) | Best effort |

## Reporting a Vulnerability

Please do not open public issues for security vulnerabilities.

Use GitHub private vulnerability reporting:

1. Go to `Security` in the repository.
2. Click `Report a vulnerability`.
3. Provide a clear report including:
   - affected version/commit
   - impact
   - reproducible steps or proof of concept
   - proposed mitigation (if available)

If private reporting is unavailable in your GitHub view, open a normal issue only
with minimal information and include `[SECURITY]` in the title. A maintainer will
move the discussion to a private channel.

## Response Targets

- Initial acknowledgment: within 72 hours
- Triage/update: within 7 days
- Fix timeline: depends on severity and reproducibility

## Disclosure Policy

- We prefer coordinated disclosure.
- After a fix is available, we may publish a short advisory in release notes.
- Credit is given on request unless anonymity is requested.

## Scope

In scope:

- vulnerabilities in `fnvision/` runtime code
- vulnerabilities in bundled tooling under `fnvision/tools/`

Out of scope:

- vulnerabilities in third-party dependencies themselves
- local environment/package-manager misconfiguration
