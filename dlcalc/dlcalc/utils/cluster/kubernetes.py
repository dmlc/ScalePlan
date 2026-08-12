from __future__ import annotations

import dataclasses
import json
import os
import re

import kubernetes  # type: ignore[import-untyped]
from kubernetes.client import CoreV1Api

pod_id_pattern = re.compile(r"worker-(\d+)")

# Kubernetes annotation holding a {csi_driver_name: node_id} map. Which driver
# is present depends on what the cluster has installed, so we resolve the entry
# instead of assuming a particular one. Set DLCALC_CSI_DRIVER to disambiguate
# when a node registers more than one driver.
NODE_ID_ANNOTATION = "csi.volume.kubernetes.io/nodeid"
CSI_DRIVER_ENV_VAR = "DLCALC_CSI_DRIVER"


def _node_instance_id(annotations: dict[str, str], node_name: str) -> str:
    """Resolve a node's cloud instance ID from its CSI nodeid annotation."""
    if NODE_ID_ANNOTATION not in annotations:
        raise RuntimeError(
            f"node {node_name} has no {NODE_ID_ANNOTATION} annotation; "
            "cannot determine its instance ID."
        )

    driver_to_node_id = json.loads(annotations[NODE_ID_ANNOTATION])

    configured_driver = os.environ.get(CSI_DRIVER_ENV_VAR)
    if configured_driver is not None:
        if configured_driver not in driver_to_node_id:
            raise RuntimeError(
                f"{CSI_DRIVER_ENV_VAR}={configured_driver} is not registered on node "
                f"{node_name}. Available drivers: {sorted(driver_to_node_id)}"
            )
        return str(driver_to_node_id[configured_driver])

    if len(driver_to_node_id) != 1:
        raise RuntimeError(
            f"node {node_name} registers {len(driver_to_node_id)} CSI drivers "
            f"({sorted(driver_to_node_id)}); set {CSI_DRIVER_ENV_VAR} to pick one."
        )

    return str(next(iter(driver_to_node_id.values())))


@dataclasses.dataclass
class KubernetesJobMember:
    pod_name: str
    worker_id: int

    node_instance_type: str
    node_instance_id: str
    node_region: str
    node_az: str


def get_kubernetes_cluster_members(job_search_prefix: str) -> list[KubernetesJobMember]:
    kubernetes.config.load_kube_config()  # type: ignore[attr-defined]
    client = CoreV1Api()

    node_name_to_pod_name = {}
    for pod_info in client.list_namespaced_pod(namespace="default", watch=False).items:
        pod_name: str = pod_info.metadata.name if pod_info.metadata else ""  # type: ignore[union-attr,assignment]
        node_name: str = pod_info.spec.node_name if pod_info.spec else ""  # type: ignore[union-attr,assignment]

        if not pod_name.startswith(job_search_prefix):
            continue

        node_name_to_pod_name[node_name] = pod_name

    cluster_regions = set()
    cluster_azs = set()
    cluster_instance_types = set()

    cluster_members = []
    for node_info in client.list_node().items:
        if not node_info.metadata:
            continue
        node_name = node_info.metadata.name  # type: ignore[union-attr,assignment]
        if not node_info.metadata.annotations:  # type: ignore[union-attr]
            continue
        node_instance_id = _node_instance_id(node_info.metadata.annotations, node_name)  # type: ignore[union-attr,arg-type]
        if node_name in node_name_to_pod_name:
            pod_name = node_name_to_pod_name[node_name]
            match = re.search(pod_id_pattern, pod_name)
            if match is None:
                raise ValueError(f"Pod name {pod_name} doesn't match pattern {pod_id_pattern}")
            worker_id = int(match.group(1))

            node_region = (
                node_info.metadata.labels["topology.kubernetes.io/region"]
                if node_info.metadata.labels
                else ""
            )  # type: ignore[union-attr,index]
            cluster_regions.add(node_region)

            node_az = (
                node_info.metadata.labels["topology.kubernetes.io/zone"]
                if node_info.metadata.labels
                else ""
            )  # type: ignore[union-attr,index]
            cluster_azs.add(node_az)

            node_instance_type = (
                node_info.metadata.labels["node.kubernetes.io/instance-type"]
                if node_info.metadata.labels
                else ""
            )  # type: ignore[union-attr,index]
            cluster_instance_types.add(node_instance_type)

            member_def = KubernetesJobMember(
                pod_name=pod_name,
                worker_id=worker_id,
                node_instance_type=node_instance_type,
                node_instance_id=node_instance_id,
                node_region=node_region,
                node_az=node_az,
            )
            cluster_members.append(member_def)

    if len(cluster_regions) != 1:
        raise RuntimeError(f"found {len(cluster_regions)} regions but only expected 1.")

    if len(cluster_instance_types) != 1:
        print(f"WARNING: found {len(cluster_instance_types)}, generally there should only be 1.")

    if len(cluster_members) != len(node_name_to_pod_name):
        raise RuntimeError(
            f"found {len(node_name_to_pod_name)} pods but "
            f"was only able to match {len(cluster_members)} to instance IDs"
        )

    return cluster_members


def get_free_instances() -> set[str]:
    kubernetes.config.load_kube_config()  # type: ignore[attr-defined]
    client = CoreV1Api()

    occupied_nodes = set()
    for pod_info in client.list_namespaced_pod(namespace="default", watch=False).items:
        if pod_info.spec:
            occupied_nodes.add(pod_info.spec.node_name)  # type: ignore[union-attr]

    free_instance_ids = set()
    for node_info in client.list_node().items:
        if not node_info.metadata:
            continue
        node_name = node_info.metadata.name  # type: ignore[union-attr,assignment]
        if node_name not in occupied_nodes:
            if not node_info.metadata.annotations:  # type: ignore[union-attr]
                continue
            node_instance_id = _node_instance_id(node_info.metadata.annotations, node_name)  # type: ignore[union-attr,arg-type]
            free_instance_ids.add(node_instance_id)

    return free_instance_ids
