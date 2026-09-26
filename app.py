import os
import re
import sys
import math
import socket
import collections
from typing import List, Dict, Tuple, Any

try:
    import gradio as gr
except ImportError:
    raise ImportError("Gradio is required. Please run: pip install gradio")

# ==============================================================================
# 1. PURE-PYTHON VECTOR RAG ENGINE (<35MB RAM Footprint)
# ==============================================================================

class PurePythonVectorRAG:
    """Lightweight TF-IDF Vector RAG Engine over My Oracle Support (MOS) Knowledge Base."""
    def __init__(self, documents: List[Dict[str, str]]):
        self.documents = documents
        self.doc_tokens = []
        self.vocabulary = set()
        self.idf = {}
        self.doc_vectors = []
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b[a-zA-Z0-9_\-\.]{2,}\b', text.lower())

    def _build_index(self):
        for doc in self.documents:
            tokens = self._tokenize(doc["content"] + " " + doc["metadata"])
            self.doc_tokens.append(tokens)
            self.vocabulary.update(tokens)

        n_docs = len(self.documents)
        df = collections.defaultdict(int)
        for tokens in self.doc_tokens:
            for token in set(tokens):
                df[token] += 1

        for token, freq in df.items():
            self.idf[token] = math.log((n_docs + 1) / (freq + 1)) + 1.0

        for tokens in self.doc_tokens:
            self.doc_vectors.append(self._vectorize(tokens))

    def _vectorize(self, tokens: List[str]) -> Dict[str, float]:
        tf = collections.defaultdict(int)
        for t in tokens:
            tf[t] += 1
        vec = {t: tf[t] * self.idf.get(t, 1.0) for t in tf if t in self.idf}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm > 0:
            for t in vec:
                vec[t] /= norm
        return vec

    def _cosine_similarity(self, v1: Dict[str, float], v2: Dict[str, float]) -> float:
        return sum(v1[k] * v2.get(k, 0.0) for k in v1)

    def search(self, query: str, top_k: int = 2) -> List[Tuple[Dict[str, str], float]]:
        q_tokens = self._tokenize(query)
        q_vec = self._vectorize(q_tokens)
        scores = []
        for idx, d_vec in enumerate(self.doc_vectors):
            score = self._cosine_similarity(q_vec, d_vec)
            scores.append((self.documents[idx], score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

# Official My Oracle Support (MOS) Knowledge Corpus
MOS_KNOWLEDGE_BASE = [
    {
        "doc_id": "Doc ID 1388147.1",
        "title": "Troubleshooting ORA-04031: Unable to Allocate Shared Memory in Shared Pool",
        "metadata": "ORA-04031 memory shared_pool SGA 4031 allocation fragmentation alter system flush shared_pool",
        "content": "ORA-04031 indicates Oracle cannot allocate contiguous bytes in the shared pool or large pool. Causes: unshared SQL parsing, missing bind variables, or insufficient SHARED_POOL_SIZE. Action: Flush shared pool (ALTER SYSTEM FLUSH SHARED_POOL), evaluate cursor_sharing=FORCE, and scale SGA target memory parameters by 25%."
    },
    {
        "doc_id": "Doc ID 60.1",
        "title": "Resolution Matrix for ORA-00060: Deadlock Detected While Waiting for Resource",
        "metadata": "ORA-00060 deadlock ORA-60 TX ITL index unindexed foreign key lock wait",
        "content": "ORA-00060 occurs when transactions hold mutually conflicting row/table locks. Trace analysis identifies missing foreign key indexes on child tables causing full table share lock escalations. Support Protocol: stage SQL profile, isolate breaker session, and notify application team to add child table index."
    },
    {
        "doc_id": "Doc ID 2831411.1",
        "title": "OCI FastConnect BGP Peering Down Diagnostic Guide",
        "metadata": "OCI FastConnect BGP flapping ASN cross-connect virtual circuit peering down drg",
        "content": "FastConnect BGP down events stem from ASN mismatch, MD5 auth key discrepancy, or interface MTU mismatch (1500 vs 9000 bytes) between Customer Edge (CE) and Oracle DRG. Protocol: verify redundant virtual circuit failover state, test DRG route tables, and trigger circuit loopback handshake."
    },
    {
        "doc_id": "Doc ID 2490182.1",
        "title": "Autonomous Database (ATP/ADW) Auto-Scaling Quota and CPU Throttling",
        "metadata": "Autonomous Database ATP ADW quota OCPU scaling resource locked max_ocpu ucm",
        "content": "Autonomous Database halts auto-scaling when tenancy compartment limits or Universal Credit (UCM) soft quotas are hit. Remediate by applying an automated OCPU quota expansion waiver, updating dynamic scaling ceiling, and re-allocating idle capacity across compartment compute pools."
    },
    {
        "doc_id": "Doc ID 1982731.1",
        "title": "Data Guard Physical Standby Transport and Apply Lag Synchronization",
        "metadata": "Data Guard standby lag ORA-16198 redo transport desync broker failover",
        "content": "Physical Standby desynchronization occurs during network packet drops or archive sequence gaps. Protocol: query V$ARCHIVE_GAP, stage missing RMAN archivelog bundles, restart standby apply service (ALTER DATABASE RECOVER MANAGED STANDBY DATABASE DISCONNECT), and confirm MRP0 active state."
    },
    {
        "doc_id": "Doc ID 2109845.1",
        "title": "Oracle GoldenGate Microservices Replicat Abend and Trail Desync",
        "metadata": "GoldenGate replicat abend ORA-00001 unique constraint checkpoint trail file lag",
        "content": "GoldenGate Replicat abends typically due to schema drift or unique constraint violations on target tables. Remediation: engage conflict detection and resolution (CDR), isolate conflicting transaction trail offset, execute discard file dump, and advance replicat sequence."
    },
    {
        "doc_id": "Doc ID 2701449.1",
        "title": "Exadata Cell Flash Cache Read Timeout and Smart Scan Degradation",
        "metadata": "Exadata cell smart scan flash cache latency storage server cellcli",
        "content": "Cell storage flash cache timeouts occur during high unindexed physical reads or degraded NVMe flash disks. Protocol: run CellCLI 'list celldisk where errorCount > 0', stage temporary I/O resource management (IORM) objective to high-throughput, and failover I/O to mirror partner grid disk."
    },
    {
        "doc_id": "Doc ID 2601944.1",
        "title": "OCI IAM SAML 2.0 Identity Provider Federation Dropout",
        "metadata": "OCI IAM SAML identity provider federation cert expired 401 unauthorized",
        "content": "SSO federation authentication drops when the IdP SAML signing certificate expires or clock skew exceeds 300 seconds. Protocol: stage temporary direct break-glass IAM administrator bypass and re-import signing XML metadata."
    }
]

mos_rag_engine = PurePythonVectorRAG(MOS_KNOWLEDGE_BASE)

# ==============================================================================
# 2. AUTONOMOUS ReAct TOOLS (Simulated OCI & MOS APIs)
# ==============================================================================

def tool_inspect_oci_telemetry(resource_ocid: str, region: str) -> Dict[str, Any]:
    """Inspects live telemetry metrics across OCI Compute, Exadata, or Autonomous Database."""
    telemetry_db = {
        "ocid1.autonomousdb.oc1.iad.abuwx4": {
            "resource_name": "PROD-ATP-FINANCE-01",
            "cpu_utilization": "98.4%",
            "sga_shared_pool_free_mb": 42.1,
            "memory_pressure": "CRITICAL",
            "active_sessions": 340,
            "health_status": "DEGRADED"
        },
        "ocid1.database.oc1.phx.deadlock9": {
            "resource_name": "EXADATA-CORE-RETAIL",
            "cpu_utilization": "42.0%",
            "lock_waits": "DETECTED (Row-exclusive TX on Table ORDERS)",
            "deadlock_victim_sid": "SID 842 / Serial 1902",
            "health_status": "INCIDENT_PENDING"
        },
        "ocid1.virtualcircuit.oc1.iad.fastconnect": {
            "resource_name": "FC-US-EAST-PRIMARY",
            "bgp_status": "DOWN",
            "redundant_circuit_status": "UP (Carrying 100% Traffic)",
            "packet_loss": "12.4%",
            "health_status": "FAILOVER_ACTIVE"
        },
        "ocid1.autonomousdb.oc1.fra.scaling": {
            "resource_name": "CORE-ADW-REPORTING",
            "cpu_utilization": "100%",
            "current_ocpu": 8,
            "max_configured_ocpu": 24,
            "compartment_quota_cap": 8,
            "health_status": "THROTTLED"
        },
        "ocid1.database.oc1.phx.standby01": {
            "resource_name": "DATAGUARD-DR-FINANCE",
            "cpu_utilization": "31.2%",
            "mrp0_status": "STOPPED",
            "transport_lag_mins": 142,
            "health_status": "DESYNCHRONIZED"
        },
        "ocid1.goldengate.oc1.iad.replicat": {
            "resource_name": "OGG-PAYMENTS-REP",
            "cpu_utilization": "15.0%",
            "replicat_status": "ABENDED",
            "trail_lag_sec": 7820,
            "health_status": "HALTED"
        },
        "ocid1.exadata.oc1.iad.cell01": {
            "resource_name": "EXADATA-STORAGE-CELL-03",
            "cpu_utilization": "89.1%",
            "flash_cache_hit_rate": "41.2%",
            "iorm_mode": "BASIC",
            "health_status": "I/O_DEGRADED"
        },
        "ocid1.idp.oc1.iad.federation": {
            "resource_name": "ENTERPRISE-OKTA-IDP",
            "cpu_utilization": "N/A",
            "saml_cert_expiry": "EXPIRED (0 days)",
            "clock_skew_sec": 4,
            "health_status": "AUTH_FAILED"
        }
    }
    return telemetry_db.get(resource_ocid, {
        "resource_name": "GENERIC-OCI-INSTANCE",
        "cpu_utilization": "68.5%",
        "health_status": "HEALTHY",
        "region": region
    })

def tool_query_mos_bugdb(error_signature: str, db_version: str) -> Dict[str, Any]:
    """Queries Oracle internal BugDB and Release Update (RU) repositories."""
    bug_records = {
        "ORA-04031": {
            "bug_id": "BUG-32910482",
            "severity": "Sev-1",
            "known_patch": "Patch 34789012 (Interim One-Off RU 19.18)",
            "workaround": "ALTER SYSTEM FLUSH SHARED_POOL; Set _kghfds_fragmentation_guard=TRUE",
            "reproducible": "Yes, high concurrent cursor parsing without bind variables"
        },
        "ORA-00060": {
            "bug_id": "BUG-29810311",
            "severity": "Sev-2",
            "known_patch": "Not a database defect (Application Transaction Lock Contention)",
            "workaround": "Create missing foreign key B-Tree index on FK_CUSTOMER_ID to prevent table locks",
            "reproducible": "Application thread lock race"
        },
        "FASTCONNECT_BGP_DOWN": {
            "bug_id": "OCI-NET-89104",
            "severity": "Sev-1",
            "known_patch": "DRG Firmware Update Patch 2.4",
            "workaround": "Force BGP keepalive timer reset & renegotiate MTU handshake to 9000 bytes",
            "reproducible": "Yes, CE edge router configuration drift"
        },
        "AUTONOMOUS_QUOTA_LOCKED": {
            "bug_id": "OCI-UCM-44910",
            "severity": "Sev-2",
            "known_patch": "Autonomous CP Service Patch v23.4",
            "workaround": "Issue automated IAM compartment quota ceiling override token",
            "reproducible": "Tenancy soft limit threshold reached"
        },
        "ORA-16198": {
            "bug_id": "BUG-31094811",
            "severity": "Sev-2",
            "known_patch": "Data Guard Broker Patch RU 19.16",
            "workaround": "Fetch missing archive sequences 89201-89240 and restart managed recovery",
            "reproducible": "Transient WAN network drop"
        },
        "ORA-00001": {
            "bug_id": "OGG-BUG-489102",
            "severity": "Sev-2",
            "known_patch": "GoldenGate Patch 23.3.0.1",
            "workaround": "Enable Conflict Detection and Resolution (CDR) or discard offending record",
            "reproducible": "Target table manual insert collision"
        },
        "EXADATA_FLASH_TIMEOUT": {
            "bug_id": "EXA-BUG-982103",
            "severity": "Sev-1",
            "known_patch": "Exadata System Software 22.1.8.0.0",
            "workaround": "Switch IORM objective to AUTO and disable degraded flash cell slot",
            "reproducible": "Unindexed full table scans saturated NVMe channels"
        },
        "IAM_SAML_EXPIRED": {
            "bug_id": "OCI-IAM-11029",
            "severity": "Sev-1",
            "known_patch": "OCI Identity Control Plane v24.1",
            "workaround": "Issue emergency direct console admin session and upload renewed metadata",
            "reproducible": "Annual certificate rotation expiration"
        }
    }
    for key, data in bug_records.items():
        if key in error_signature.upper() or error_signature.upper() in key:
            return data
    return {
        "bug_id": "BUG-GENERIC-ORACLE-LOG",
        "severity": "Sev-3",
        "known_patch": "Apply Latest Database Release Update (RU)",
        "workaround": "Collect TFA (Trace File Analyzer) bundle and forward to Tier-3 PSE",
        "reproducible": "Unknown"
    }

def tool_verify_ucm_and_sla(tenancy_ocid: str, sev_tier: str, support_tier: str) -> Dict[str, Any]:
    """Audits Universal Credits Model (UCM) entitlement and Sev-1 SLA countdown."""
    is_sev1 = (sev_tier == "Sev-1")
    irt = "15 Minutes" if is_sev1 else ("1 Hour" if sev_tier == "Sev-2" else "4 Hours")
    penalty = "$15,000/hr in potential service credits" if (is_sev1 and support_tier == "Premier Enterprise Support") else "$2,500/hr"
    return {
        "tenancy_ocid": tenancy_ocid,
        "support_tier": support_tier,
        "contractual_irt_sla": irt,
        "sla_credit_risk": f"HIGH ({penalty})" if is_sev1 else "LOW (Nominal Risk)",
        "autonomous_waiver_eligible": True
    }

def tool_execute_oci_remediation(action_type: str, resource_ocid: str) -> Dict[str, Any]:
    """Simulates real-time remediation execution on live OCI infrastructure."""
    remediation_catalog = {
        "FLUSH_SHARED_POOL_AND_STAGE_PATCH": {
            "status": "EXECUTED_SUCCESSFULLY",
            "action": "Issued ALTER SYSTEM FLUSH SHARED_POOL on resource. Staged One-off Patch 34789012 in OCI Patch Manager.",
            "duration_ms": 410
        },
        "ISOLATE_DEADLOCK_TRANSACTION": {
            "status": "EXECUTED_SUCCESSFULLY",
            "action": "Rolled back blocking breaker session SID 842. Generated foreign key indexing advisory for ORDERS table.",
            "duration_ms": 180
        },
        "RENEGOTIATE_BGP_AND_FAILOVER": {
            "status": "EXECUTED_SUCCESSFULLY",
            "action": "Triggered BGP keepalive reset on DRG. Confirmed 100% traffic reroute to secondary virtual circuit.",
            "duration_ms": 290
        },
        "EXPAND_AUTONOMOUS_OCPU_QUOTA": {
            "status": "EXECUTED_SUCCESSFULLY",
            "action": "Applied 24-hour temporary OCPU expansion waiver from 8 to 24 OCPUs in compartment.",
            "duration_ms": 230
        },
        "RESYNC_STANDBY_DATAGUARD": {
            "status": "EXECUTED_SUCCESSFULLY",
            "action": "Pulled missing archive logs 89201-89240 via RMAN channel and restarted MRP0 managed recovery.",
            "duration_ms": 620
        },
        "ADVANCE_GOLDENGATE_REPLICAT": {
            "status": "EXECUTED_SUCCESSFULLY",
            "action": "Isolated duplicate key conflict to discard file and advanced Replicat checkpoint past trail sequence.",
            "duration_ms": 340
        },
        "OPTIMIZE_EXADATA_IORM": {
            "status": "EXECUTED_SUCCESSFULLY",
            "action": "Modified CellCLI IORM objective to HIGH_THROUGHPUT and isolated degraded flash cell 03.",
            "duration_ms": 470
        },
        "BYPASS_SAML_PROVISION_ADMIN": {
            "status": "EXECUTED_SUCCESSFULLY",
            "action": "Provisioned emergency 4-hour direct IAM break-glass token for tenancy administration.",
            "duration_ms": 150
        }
    }
    return remediation_catalog.get(action_type, {
        "status": "MANUAL_DISPATCH_REQUIRED",
        "action": f"Diagnostic runbook for {action_type} dispatched to Principal Support Engineer queue.",
        "duration_ms": 95
    })

# ==============================================================================
# 3. REASONING & ORCHESTRATION PIPELINE
# ==============================================================================

def run_oracle_copilot(
    ticket_id: str,
    tenancy_ocid: str,
    resource_ocid: str,
    region: str,
    db_version: str,
    sev_tier: str,
    support_tier: str,
    error_signature: str,
    incident_raw_log: str,
    manual_override: str
) -> Tuple[str, str, str, str]:
    """Orchestrates RAG retrieval, ReAct reasoning, and Minto work order formulation."""

    # Stage 1: Domain RAG Retrieval over MOS Knowledge Base
    search_query = f"{error_signature} {incident_raw_log}"
    retrieved_docs = mos_rag_engine.search(search_query, top_k=2)
    rag_context = ""
    for doc, score in retrieved_docs:
        rag_context += f"• [{doc['doc_id']}] {doc['title']} (Cosine Match: {score:.3f})\n  Content: {doc['content']}\n\n"

    # Stage 2: Autonomous Tool Calling
    telemetry_res = tool_inspect_oci_telemetry(resource_ocid, region)
    bugdb_res = tool_query_mos_bugdb(error_signature, db_version)
    sla_res = tool_verify_ucm_and_sla(tenancy_ocid, sev_tier, support_tier)

    # Determine automated operational remediation
    if manual_override and manual_override != "Auto-Detect Best Action":
        remediation_key = manual_override
    else:
        combined_text = (error_signature + " " + incident_raw_log).upper()
        if "04031" in combined_text or "SHARED_POOL" in combined_text:
            remediation_key = "FLUSH_SHARED_POOL_AND_STAGE_PATCH"
        elif "00060" in combined_text or "DEADLOCK" in combined_text:
            remediation_key = "ISOLATE_DEADLOCK_TRANSACTION"
        elif "BGP" in combined_text or "FASTCONNECT" in combined_text:
            remediation_key = "RENEGOTIATE_BGP_AND_FAILOVER"
        elif "QUOTA" in combined_text or "AUTONOMOUS" in combined_text:
            remediation_key = "EXPAND_AUTONOMOUS_OCPU_QUOTA"
        elif "16198" in combined_text or "STANDBY" in combined_text or "DATAGUARD" in combined_text:
            remediation_key = "RESYNC_STANDBY_DATAGUARD"
        elif "00001" in combined_text or "GOLDENGATE" in combined_text or "REPLICAT" in combined_text:
            remediation_key = "ADVANCE_GOLDENGATE_REPLICAT"
        elif "FLASH" in combined_text or "EXADATA" in combined_text or "TIMEOUT" in combined_text:
            remediation_key = "OPTIMIZE_EXADATA_IORM"
        elif "SAML" in combined_text or "FEDERATION" in combined_text or "CERT" in combined_text:
            remediation_key = "BYPASS_SAML_PROVISION_ADMIN"
        else:
            remediation_key = "MANUAL"

    remediation_res = tool_execute_oci_remediation(remediation_key, resource_ocid)

    # Stage 3: Transparent ReAct Trace
    react_trace = f"""[AGENT PERCEPTION]:
- Inbound Incident: {ticket_id} ({sev_tier}) | Tier: {support_tier} | Region: {region}
- Tenancy OCID: {tenancy_ocid}
- Target Resource: {resource_ocid}
- Error Pattern: {error_signature} | Platform: Oracle Database / OCI ({db_version})

[THOUGHT - STEP 1]: Audit live OCI resource telemetry and memory/node saturation levels.
-> ACTION: tool_inspect_oci_telemetry('{resource_ocid}', '{region}')
-> OBSERVATION: Resource {telemetry_res.get('resource_name')}, CPU: {telemetry_res.get('cpu_utilization')}, Health: {telemetry_res.get('health_status')}.

[THOUGHT - STEP 2]: Match error signature against internal BugDB records and validated MOS Knowledge Base.
-> ACTION: tool_query_mos_bugdb('{error_signature}', '{db_version}')
-> OBSERVATION: Matched Bug Record: {bugdb_res.get('bug_id')} | Known Patch: {bugdb_res.get('known_patch')}. Workaround: {bugdb_res.get('workaround')}.

[THOUGHT - STEP 3]: Check contractual SLA thresholds and Universal Credit Model (UCM) credit exposure.
-> ACTION: tool_verify_ucm_and_sla('{tenancy_ocid}', '{sev_tier}', '{support_tier}')
-> OBSERVATION: IRT SLA is {sla_res.get('contractual_irt_sla')}. Financial credit exposure: {sla_res.get('sla_credit_risk')}.

[THOUGHT - STEP 4]: Execute operational containment on live OCI infrastructure to mitigate outage.
-> ACTION: tool_execute_oci_remediation('{remediation_key}', '{resource_ocid}')
-> OBSERVATION: Execution Status: {remediation_res.get('status')} | {remediation_res.get('action')} (Latency: {remediation_res.get('duration_ms')}ms).
"""

    # Stage 4: Formulate Minto PSE Work Order
    pse_work_order = f"""### EXECUTIVE SUMMARY (MINTO PYRAMID ROOT CAUSE & ACTION)
1. **Immediate Situation:** {sev_tier} incident on **{telemetry_res.get('resource_name')}** ({region}) caused by **{error_signature}**. Automated stabilization completed in **{remediation_res.get('duration_ms')}ms**.
2. **Underlying Defect:** Validated against internal BugDB record **{bugdb_res.get('bug_id')}**. Telemetry confirmed state: `{telemetry_res.get('health_status')}`.
3. **Required PSE Actions:**
   - Review MOS Document **{retrieved_docs[0][0]['doc_id']}**: *{retrieved_docs[0][0]['title']}*.
   - Stage Interim Patch **{bugdb_res.get('known_patch')}** in client tenancy staging bucket.
   - Execute permanent corrective runbook: `{bugdb_res.get('workaround')}`.
4. **SLA Governance:** Contractual Initial Response Time ({sla_res.get('contractual_irt_sla')}) safeguarded. Risk of SLA credit penalty eliminated.
"""

    # Stage 5: Formulate Customer-Ready MOS SR Update
    customer_ready_update = f"""Dear Oracle Cloud Customer,

Our Oracle Automated Telemetry & Support Copilot has triaged the incident reported under Service Request #{ticket_id}.

• Status Summary: Automated diagnostic containment has been completed on your instance ({telemetry_res.get('resource_name')}).
• Diagnostic Finding: The system identified an occurrence matching signature '{error_signature}'. Our automated remediation agent executed:
  -> {remediation_res.get('action')}
• Recommended Customer Action: Please review My Oracle Support Knowledge Base Document:
  -> {retrieved_docs[0][0]['doc_id']}: {retrieved_docs[0][0]['title']}
• Next Steps: A Principal Support Engineer (PSE) has been assigned and is verifying the staging of {bugdb_res.get('known_patch')}. Your support SLA is actively monitored.

Best regards,  
Oracle Cloud & Database Global Support Services
"""

    return react_trace, rag_context, pse_work_order, customer_ready_update

# ==============================================================================
# 4. PRESET INCIDENT SCENARIOS & QUICK EXAMPLES
# ==============================================================================

PRESET_SCENARIOS = {
    "1. ORA-04031 Shared Pool Exhaustion (Prod ATP)": (
        "SR-3-99481029",
        "ocid1.tenancy.oc1..aaaaaaaatargettenancy01",
        "ocid1.autonomousdb.oc1.iad.abuwx4",
        "us-ashburn-1",
        "Oracle Database 19c RU 19.17",
        "Sev-1",
        "Premier Enterprise Support",
        "ORA-04031",
        "alert.log: ORA-04031: unable to allocate 4096 bytes of shared memory [kghsseg: kghssp] [shared pool] [SQLA^4a810]."
    ),
    "2. ORA-00060 Deadlock on Exadata Retail Orders": (
        "SR-3-88219401",
        "ocid1.tenancy.oc1..aaaaaaaaretailglobal",
        "ocid1.database.oc1.phx.deadlock9",
        "us-phoenix-1",
        "Oracle Exadata X9M (DB 21c)",
        "Sev-1",
        "Premier Enterprise Support",
        "ORA-00060",
        "alert.log: ORA-00060: Deadlock detected while waiting for resource. Resource Name: TX-0004001c-000084fa. Table: ORDERS."
    ),
    "3. OCI FastConnect BGP Peering Down (DRG)": (
        "SR-3-77291044",
        "ocid1.tenancy.oc1..aaaaaaaabankingcorp",
        "ocid1.virtualcircuit.oc1.iad.fastconnect",
        "us-ashburn-1",
        "OCI Network Fabric 2026.1",
        "Sev-1",
        "Premier Enterprise Support",
        "FASTCONNECT_BGP_DOWN",
        "OCI Metrics: BGP session state transitioned to DOWN on Virtual Circuit FC-US-EAST-PRIMARY. AS 31898 keepalive expired."
    ),
    "4. Autonomous Data Warehouse Scaling Quota Lock": (
        "SR-3-66291083",
        "ocid1.tenancy.oc1..aaaaaaaafintechlab",
        "ocid1.autonomousdb.oc1.fra.scaling",
        "eu-frankfurt-1",
        "Autonomous Data Warehouse Serverless",
        "Sev-2",
        "Premier Enterprise Support",
        "AUTONOMOUS_QUOTA_LOCKED",
        "Console Alarm: Auto-scaling request to 24 OCPUs rejected. Tenancy compartment limit reached: 8/8 OCPUs allocated."
    ),
    "5. Data Guard Standby Redo Apply Lag": (
        "SR-3-55102948",
        "ocid1.tenancy.oc1..aaaaaaaatelecom01",
        "ocid1.database.oc1.phx.standby01",
        "us-phoenix-1",
        "Oracle Database 19c Enterprise",
        "Sev-2",
        "Extended Support",
        "ORA-16198",
        "alert.log: ORA-16198: Timeout incurred on network IO during redo transport. Gap sequence detected from thread 1 seq 89201 to 89240."
    ),
    "6. GoldenGate Replicat Abend (Unique Constraint)": (
        "SR-3-44102987",
        "ocid1.tenancy.oc1..aaaaaaaapaymentgateway",
        "ocid1.goldengate.oc1.iad.replicat",
        "us-ashburn-1",
        "GoldenGate Microservices 23c",
        "Sev-2",
        "Premier Enterprise Support",
        "ORA-00001",
        "ggserr.log: WARNING OGG-00869 OCI Error ORA-00001: unique constraint (PAYMENTS.PK_TRANSACTION) violated. Replicat abended."
    ),
    "7. Exadata Cell Flash Cache Read Timeout": (
        "SR-3-33104921",
        "ocid1.tenancy.oc1..aaaaaaaafinancesec",
        "ocid1.exadata.oc1.iad.cell01",
        "us-ashburn-1",
        "Exadata System Software 22.1",
        "Sev-1",
        "Premier Enterprise Support",
        "EXADATA_FLASH_TIMEOUT",
        "cellcli: ALERT: Flash cache read timeout exceeding 5000ms on Storage Server 03. High physical unindexed reads."
    ),
    "8. OCI IAM SAML 2.0 Federation Certificate Expired": (
        "SR-3-22019483",
        "ocid1.tenancy.oc1..aaaaaaaaglobalhealth",
        "ocid1.idp.oc1.iad.federation",
        "us-ashburn-1",
        "OCI Identity Domains 2026",
        "Sev-1",
        "Premier Enterprise Support",
        "IAM_SAML_EXPIRED",
        "IAM Security Log: Inbound SAML 2.0 assertion rejected from Okta IdP. Error: Signing certificate expired at 00:00 UTC."
    )
}

DEFAULT_PROMPT_EXAMPLES = [
    ["SR-3-99481029", "ORA-04031", "alert.log: ORA-04031: unable to allocate 4096 bytes of shared memory [kghsseg: kghssp] [shared pool] [SQLA^4a810]."],
    ["SR-3-88219401", "ORA-00060", "alert.log: ORA-00060: Deadlock detected while waiting for resource on Table ORDERS. Blocked session SID 842."],
    ["SR-3-77291044", "FASTCONNECT_BGP_DOWN", "OCI Metrics: BGP session state transitioned to DOWN on Virtual Circuit FC-US-EAST-PRIMARY."],
    ["SR-3-33104921", "EXADATA_FLASH_TIMEOUT", "cellcli: ALERT: Flash cache read timeout exceeding 5000ms on Storage Server 03. High unindexed table scan."],
    ["SR-3-22019483", "IAM_SAML_EXPIRED", "IAM Security Log: Inbound SAML assertion rejected from Okta IdP. Signing certificate expired."]
]

# ==============================================================================
# 5. GRADIO OPERATIONS COCKPIT
# ==============================================================================

def load_preset(scenario_name: str):
    p = PRESET_SCENARIOS[scenario_name]
    return p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]

def fill_example(ticket_id, error_sig, raw_log):
    return ticket_id, error_sig, raw_log

custom_theme = gr.themes.Soft(
    primary_hue="red",  # Oracle Corporate Red
    neutral_hue="slate"
)

with gr.Blocks(theme=custom_theme, title="Oracle Cloud & Database Agentic Service Copilot") as demo:
    gr.Markdown("""
    # 🔴 Oracle Cloud Infrastructure & Database Agentic Service Copilot
    ### Autonomous Triage, Log Diagnostics, and Real-Time Remediation for OCI & My Oracle Support (MOS)
    """)

    with gr.Row():
        with gr.Column(scale=4):
            gr.Markdown("### 1. Inbound Incident Context & Telemetry Payload")
            scenario_selector = gr.Dropdown(
                choices=list(PRESET_SCENARIOS.keys()),
                value=list(PRESET_SCENARIOS.keys())[0],
                label="⚡ Load Production Incident Scenario"
            )

            ticket_id_in = gr.Textbox(label="MOS Service Request (SR #)", value="SR-3-99481029")
            with gr.Row():
                sev_tier_in = gr.Dropdown(choices=["Sev-1", "Sev-2", "Sev-3", "Sev-4"], value="Sev-1", label="Severity Tier")
                support_tier_in = gr.Dropdown(choices=["Premier Enterprise Support", "Extended Support"], value="Premier Enterprise Support", label="Support Tier")

            with gr.Row():
                region_in = gr.Dropdown(choices=["us-ashburn-1", "us-phoenix-1", "eu-frankfurt-1", "ap-tokyo-1"], value="us-ashburn-1", label="OCI Region")
                db_ver_in = gr.Textbox(label="Oracle DB / OCI Version", value="Oracle Database 19c RU 19.17")

            tenancy_ocid_in = gr.Textbox(label="OCI Tenancy OCID", value="ocid1.tenancy.oc1..aaaaaaaatargettenancy01")
            resource_ocid_in = gr.Textbox(label="Target Resource OCID", value="ocid1.autonomousdb.oc1.iad.abuwx4")
            error_sig_in = gr.Textbox(label="Primary Error Signature", value="ORA-04031")

            raw_log_in = gr.TextArea(
                label="Raw alert.log / OCI Telemetry Snippet",
                value="alert.log: ORA-04031: unable to allocate 4096 bytes of shared memory [kghsseg: kghssp] [shared pool] [SQLA^4a810].",
                lines=3
            )

            override_in = gr.Dropdown(
                choices=[
                    "Auto-Detect Best Action",
                    "FLUSH_SHARED_POOL_AND_STAGE_PATCH",
                    "ISOLATE_DEADLOCK_TRANSACTION",
                    "RENEGOTIATE_BGP_AND_FAILOVER",
                    "EXPAND_AUTONOMOUS_OCPU_QUOTA",
                    "RESYNC_STANDBY_DATAGUARD",
                    "ADVANCE_GOLDENGATE_REPLICAT",
                    "OPTIMIZE_EXADATA_IORM",
                    "BYPASS_SAML_PROVISION_ADMIN"
                ],
                value="Auto-Detect Best Action",
                label="🛠️ Manual Operational Intervention Override"
            )

            triage_btn = gr.Button("🚀 Run Autonomous Agentic Triage", variant="primary")

            gr.Markdown("#### 💡 Quick Example Injection")
            gr.Examples(
                examples=DEFAULT_PROMPT_EXAMPLES,
                inputs=[ticket_id_in, error_sig_in, raw_log_in],
                label="Click any example below to quickly load into prompt boxes:"
            )

        with gr.Column(scale=6):
            gr.Markdown("### 2. Autonomous Diagnostic & Remediation Output")
            with gr.Tabs():
                with gr.TabItem("🧠 Transparent ReAct Agent Trace"):
                    trace_out = gr.TextArea(label="Perception-Thought-Action Execution Loop", lines=15)

                with gr.TabItem("📋 Minto PSE Work Order"):
                    work_order_out = gr.Markdown()

                with gr.TabItem("📚 Grounded MOS Knowledge (RAG)"):
                    rag_out = gr.TextArea(label="Retrieved My Oracle Support Doc IDs", lines=12)

                with gr.TabItem("✉️ Customer-Ready MOS SR Update"):
                    customer_out = gr.TextArea(label="Customer Communication Payload", lines=12)

    scenario_selector.change(
        fn=load_preset,
        inputs=[scenario_selector],
        outputs=[ticket_id_in, tenancy_ocid_in, resource_ocid_in, region_in, db_ver_in, sev_tier_in, support_tier_in, error_sig_in, raw_log_in]
    )

    triage_btn.click(
        fn=run_oracle_copilot,
        inputs=[ticket_id_in, tenancy_ocid_in, resource_ocid_in, region_in, db_ver_in, sev_tier_in, support_tier_in, error_sig_in, raw_log_in, override_in],
        outputs=[trace_out, rag_out, work_order_out, customer_out]
    )

# ==============================================================================
# 6. DYNAMIC PORT BINDING & MULTI-ENVIRONMENT LAUNCHER
# ==============================================================================

def find_available_port(starting_port: int = 7860, max_attempts: int = 50) -> int:
    """Scans and returns the first available TCP socket port to avoid collisions."""
    for p in range(starting_port, starting_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if s.connect_ex(('127.0.0.1', p)) != 0:
                return p
    return starting_port

if __name__ == "__main__":
    is_colab = "google.colab" in sys.modules
    env_port = os.environ.get("PORT")

    if env_port:
        port = int(env_port)
        host = "0.0.0.0"
        share = False
    elif is_colab:
        port = find_available_port(7860)
        host = "127.0.0.1"
        share = True
    else:
        port = find_available_port(7860)
        host = "127.0.0.1"
        share = False

    print(f"🚀 Oracle Agentic Copilot active: http://{host}:{port}")
    demo.launch(
        server_name=host,
        server_port=port,
        inbrowser=True,
        share=share
    )
