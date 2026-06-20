# Security Policy

## 🛡️ Supported Versions

We actively maintain and provide security updates for the latest major release of the MLOps pipeline and the mobile edge application.

| Version | Supported          | Notes |
| ------- | ------------------ | ----- |
| 1.0.x   | :white_check_mark: | Active deployment |
| < 1.0   | :x:                | Deprecated / Development builds |

## 🎯 Scope

The following components are within the scope of our security program:
* The TensorFlow model generation pipeline (`src/`).
* The React Native camera interface and JSI bindings (`mobile_app/`).
* Data privacy mechanisms ensuring zero-cloud image transmission.

## 🚨 Reporting a Vulnerability

Security and data privacy are foundational to this edge-deployment architecture. We deeply appreciate community efforts to identify vulnerabilities.

If you discover a security issue, **DO NOT open a public GitHub issue.**

Please report it directly via email to the repository owner. 

### Service Level Agreement (SLA)
Upon receiving your report, we commit to the following response timeline:
1. **Acknowledgement:** Within 48 hours.
2. **Triage & Verification:** Within 5 business days.
3. **Patch Deployment:** Depending on severity, within 14-30 days.

*Please include detailed reproduction steps, the affected component, and potential impact in your report.*