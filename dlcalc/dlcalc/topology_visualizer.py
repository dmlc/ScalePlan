# mypy: disable-error-code="import-not-found"
"""
given a kubernetes pod name prefix for some compute cluster, retrieve AWS network
topology info and plot.
"""

import re
from argparse import ArgumentParser

import boto3
from pyvis.network import Network  # type: ignore[import-untyped]

import dlcalc.utils.cluster.ec2
import dlcalc.utils.cluster.kubernetes

pod_id_pattern = re.compile(r"worker-(\d+)")


def _dict_to_label(dict: dict) -> str:
    kv_pairs = []
    for key, value in dict.items():
        kv_pairs.append(f"{key}: {value}")
    return "\n".join(kv_pairs)


def main() -> None:
    parser = ArgumentParser(__doc__)
    parser.add_argument(
        "search_prefix",
        help="kubernetes pod name prefix to use when filtering pods returned by `kubectl get pods`",
    )
    args = parser.parse_args()

    job_search_prefix = args.search_prefix

    ######################################################################################
    # STEP 1: Get the list of instance IDs involved in a kubernetes job
    ######################################################################################
    cluster_members = dlcalc.utils.cluster.kubernetes.get_kubernetes_cluster_members(
        job_search_prefix
    )

    ######################################################################################
    # STEP 2: Get topology of the instances acquired earlier.
    ######################################################################################
    instance_ids = {m.node_instance_id for m in cluster_members}
    (cluster_region,) = set(m.node_region for m in cluster_members)
    cluster_instance_types = {m.node_instance_type for m in cluster_members}
    cluster_azs = {m.node_az for m in cluster_members}
    instance_id_to_job_member = {m.node_instance_id: m for m in cluster_members}

    print(f"getting topology info for instance IDs {instance_ids}")
    ec2_client = boto3.client("ec2", region_name=cluster_region)

    net = Network(directed=True)
    for instance_info in dlcalc.utils.cluster.ec2.iter_instance_info(
        ec2_client=ec2_client,
        accepted_node_instance_types=cluster_instance_types,
        accepted_node_availability_zones=cluster_azs,
        accepted_instance_ids=instance_ids,
    ):
        job_member = instance_id_to_job_member[instance_info.instance_id]

        net.add_node(
            instance_info.instance_id,
            title=_dict_to_label(
                {
                    "PodName": job_member.pod_name,
                    "WorkerId": job_member.worker_id,
                    "InstanceType": instance_info.instance_type,
                    "AvailabilityZone": instance_info.instance_az,
                }
            ),
            color="blue",
            physics=False,
        )

        for network_node in instance_info.network_nodes:
            net.add_node(network_node, color="red", physics=False)

        # order of network nodes: farthest from instance -> closest to instance
        chain = [*instance_info.network_nodes, instance_info.instance_id]
        for edge_src_idx in range(len(chain) - 1):
            edge_dst_idx = edge_src_idx + 1
            net.add_edge(chain[edge_src_idx], chain[edge_dst_idx], physics=False)

    net.show_buttons()
    net.show("graph.html", notebook=False)


if __name__ == "__main__":
    main()
