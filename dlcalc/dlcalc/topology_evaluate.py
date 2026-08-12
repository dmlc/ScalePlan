"""
evaluates how optimal a given training job's physical topology is (in terms of
network hops in each DP ring).
"""

from argparse import ArgumentParser

import boto3  # type: ignore[import-untyped, import-not-found]

import dlcalc.utils.cluster.ec2
import dlcalc.utils.cluster.kubernetes
import dlcalc.utils.cluster.topology
from dlcalc.utils.cluster.topology import TreeNode
from dlcalc.utils.math import safe_divide


def main() -> None:
    parser = ArgumentParser(__doc__)
    parser.add_argument(
        "job_search_prefix",
        help="kubernetes pod name prefix to use when filtering pods returned by `kubectl get pods`",
    )
    parser.add_argument(
        "-t",
        "--tp-degree",
        type=int,
        required=True,
        help="tensor parallel (TP) degree",
    )
    parser.add_argument(
        "-p",
        "--pp-degree",
        type=int,
        required=True,
        help="pipeline parallel (PP) degree",
    )
    parser.add_argument(
        "-d",
        "--dp-degree",
        type=int,
        required=True,
        help="data parallel (DP) degree",
    )
    parser.add_argument(
        "-n",
        "--accelerators-per-node",
        type=int,
        default=8,
        help="number of accelerators per node",
    )
    args = parser.parse_args()
    job_search_prefix = args.job_search_prefix
    tp_degree = args.tp_degree
    pp_degree = args.pp_degree
    dp_degree = args.dp_degree
    accelerators_per_node = args.accelerators_per_node

    world_size = tp_degree * pp_degree * dp_degree
    expected_n_nodes = safe_divide(world_size, accelerators_per_node)

    assert tp_degree <= accelerators_per_node

    # we only care to optimize inter-node DP communication
    # (intra-node DP communication can't be optimized with better instance placement),
    # so we'll ignore the parts of the ring that traverse intra-node links.
    dp_ranks_per_node = safe_divide(accelerators_per_node, tp_degree)
    nodes_per_dp_ring = safe_divide(dp_degree, dp_ranks_per_node)
    print(
        f"DP={dp_degree} and {dp_ranks_per_node} DP ranks/node, "
        f"therefore there are {nodes_per_dp_ring} nodes per DP ring"
    )

    # everything we look at here is on the instance level.

    # if TP < accelerators per node, then there will be multiple DP ranks on a single
    # node. we'll ignore this intra-node DP communication as we just care about
    # improving inter-node DP communication
    # we're making the assumption that all TP groups are within a single node.
    # since we're looking at things at the instance level, there's only one
    # ring per PP rank.

    ######################################################################################
    # STEP 1: Get instance info involved in a kubernetes job
    ######################################################################################
    cluster_members = dlcalc.utils.cluster.kubernetes.get_kubernetes_cluster_members(
        job_search_prefix
    )
    if len(cluster_members) != expected_n_nodes:
        raise ValueError(
            f"with world size {world_size} "
            f"expected to find {expected_n_nodes} nodes "
            f"but found {len(cluster_members)}"
        )
    worker_id_to_instance_id = {m.worker_id: m.node_instance_id for m in cluster_members}
    (cluster_region,) = {m.node_region for m in cluster_members}
    cluster_instance_types = {m.node_instance_type for m in cluster_members}
    cluster_azs = {m.node_az for m in cluster_members}

    print(
        f"found {len(cluster_members)} matching pods, with "
        f"region: {cluster_region}, instance_types: {cluster_instance_types}, azs: {cluster_azs}"
    )

    node_id_to_node = dlcalc.utils.cluster.topology.build_instance_tree(
        ec2_client=boto3.client("ec2", region_name=cluster_region),
        accepted_node_instance_types=cluster_instance_types,
        accepted_node_availability_zones=cluster_azs,
        accepted_instance_ids=set(worker_id_to_instance_id.values()),
    )

    ######################################################################################
    # STEP 2: Determine connectivity path of DP rings
    ######################################################################################

    # find placement on physical topology
    for pp_rank in range(pp_degree):
        dp_ring_for_pp_rank: list[TreeNode] = []
        for dp_rank in range(nodes_per_dp_ring):
            worker_id = pp_rank * nodes_per_dp_ring + dp_rank
            instance_id = worker_id_to_instance_id[worker_id]
            dp_ring_for_pp_rank.append(node_id_to_node[instance_id])

        traversal_dist = dlcalc.utils.cluster.topology.traversal_distance_of_ring(
            dp_ring_for_pp_rank
        )
        print(
            f"DP ring for PP={pp_rank}: traversal distance: {traversal_dist} "
            f"-> avg traversal distance: {traversal_dist / nodes_per_dp_ring:.2f}"
        )
        print()


if __name__ == "__main__":
    main()
