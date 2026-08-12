"""
determine the best instance placement and rank assignment to ensure DP ring
communication can occur with the fewest possible network hops.
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
    parser.add_argument(
        "-i",
        "--instance-type",
        type=str,
        help="required instance type",
    )
    parser.add_argument(
        "-r",
        "--region",
        type=str,
        help="required AWS region",
    )
    parser.add_argument(
        "-a",
        "--az",
        type=str,
        help="required AWS availability zone",
    )
    args = parser.parse_args()
    tp_degree = args.tp_degree
    pp_degree = args.pp_degree
    dp_degree = args.dp_degree
    accelerators_per_node = args.accelerators_per_node

    n_required_nodes = safe_divide(tp_degree * pp_degree * dp_degree, accelerators_per_node)

    free_instance_ids = dlcalc.utils.cluster.kubernetes.get_free_instances()
    print(f"found {len(free_instance_ids)} free instances")
    instance_type = args.instance_type
    region_name = args.region
    az = args.az

    assert tp_degree <= accelerators_per_node

    # we only care to optimize inter-node DP communication
    # (intra-node DP communication can't be optimized with better instance placement),
    # so we'll ignore the parts of the ring that traverse intra-node links.
    # we're making the assumption that all TP groups are within a single node.
    # since we're looking at things at the instance level, there's only one
    # ring per PP rank.
    dp_ranks_per_node = safe_divide(accelerators_per_node, tp_degree)
    nodes_per_dp_ring = safe_divide(dp_degree, dp_ranks_per_node)
    print(
        f"DP={dp_degree} and {dp_ranks_per_node} DP ranks/node, "
        f"therefore there are {nodes_per_dp_ring} nodes per DP ring"
    )
    node_id_to_node = dlcalc.utils.cluster.topology.build_instance_tree(
        ec2_client=boto3.client("ec2", region_name=region_name),
        accepted_node_instance_types={
            instance_type,
        },
        accepted_node_availability_zones={
            az,
        },
        accepted_instance_ids=free_instance_ids,
    )
    n_discovered_instances = len(node_id_to_node)

    if n_discovered_instances < n_required_nodes:
        raise RuntimeError(
            f"training job requires {n_required_nodes} nodes but found {n_discovered_instances}"
        )
    print(f"discovered {len(node_id_to_node)} instances")

    # find root node
    (root_node,) = (node for node in node_id_to_node.values() if node.parent is None)

    # determine optimal topology
    # we want to select an ordered subset of nodes for optimal adjacency.
    # the most adjacent set of nodes for ring collectives minimizes the number
    # of network nodes that need to be crossed.
    # Physical topology is a tree. in each level we can order by the subtree size,
    # then we'll walk it DFS. this will give us an ordered list of instances that would
    # make relatively suitable rings.
    # TODO. this is sort of a heuristic, will not always find the most optimal
    # placement.
    ordered_nodes = dlcalc.utils.cluster.topology.dfs_tree_leaves_only(root_node)

    if len(ordered_nodes) != n_discovered_instances:
        # TODO. need to debug further to see why not all nodes are reachable by the DFS
        # just warning for now.
        print(
            f"[WARN] expected to traverse {n_discovered_instances} instances "
            f"but only encountered {len(ordered_nodes)}"
        )
        # raise RuntimeError(
        #     f"expected to traverse {len(instance_ids)} nodes "
        #     f"but encountered {len(ordered_nodes)}"
        # )

    ring_assignments: list[list[TreeNode]] = []
    for pp_rank in range(pp_degree):
        ring_assignments.append(ordered_nodes[:nodes_per_dp_ring])
        ordered_nodes = ordered_nodes[nodes_per_dp_ring:]

        print(f"ring for PP={pp_rank}")
        print([n.node_id for n in ring_assignments[pp_rank]])
        traversal_dist = dlcalc.utils.cluster.topology.traversal_distance_of_ring(
            ring_assignments[pp_rank]
        )
        print(
            f"traversal distance: {traversal_dist} -> avg traversal distance {traversal_dist / nodes_per_dp_ring:.2f}"
        )

    # TODO. have this output an MPI rankfile instead
    print("outputting instance list")
    for ring_assignment in ring_assignments:
        for node in ring_assignment:
            print(node.node_id)


if __name__ == "__main__":
    main()
