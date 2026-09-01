# Security Management

## Routine Antivirus Software Check

Periodically scan clusters for viruses. This protects clusters from viruses, malicious code, spyware, and malicious programs, reducing risks such as system breakdown and information leakage. Mainstream antivirus software can be used for antivirus check.

## Log Management

Pay attention to the following points in log management:

- Check whether the system can limit the size of a single log file.
- Check whether there is a mechanism for clearing logs when the log space is used up.

## Vulnerability/Function Defect Fixing

To ensure the security of the production environment and reduce the risk of attacks, periodically check the open-source communities and fix the following vulnerabilities/function defects:

- OS vulnerabilities/function defects
- Vulnerabilities/function defects in other related components

## Collective Communication Security Risk Warning

Currently, the TLS authentication function of Gloo, DataDist, and HCCL is not supported and has the following security risks:

- The default released Gloo communication library of PyTorch does not support the TLS authentication function.
- The DataDist and HCCL communication of CANN do not support the TLS authentication function.

**Risk Mitigation Measures**

- You are advised to compile and install PyTorch that supports TLS.
- Perform communication security hardening by referring to the CANN security hardening document.
- You are advised to deploy the inference service in a controlled and trusted network environment to ensure that the collective communication is in the security domain.
