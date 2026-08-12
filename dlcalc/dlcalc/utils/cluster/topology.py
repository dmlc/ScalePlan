from __future__ import annotations

import dataclasses
import functools
import itertools

from botocore.client import BaseClient  # type: ignore[import-untyped]

import dlcalc.utils.cluster.ec2


@dataclasses.dataclass
class TreeNode:
    node_id: str
    parent: TreeNode | None
    children: set[TreeNode]

    def __hash__(self) -> int:
        return hash(self.node_id)


def build_instance_tree(  # type: ignore[no-any-unimported]
    ec2_client: BaseClient,
    accepted_node_instance_types: set[str],
    accepted_node_availability_zones: set[str],
    accepted_instance_ids: set[str] | None = None,
) -> dict[str, TreeNode]:
    node_id_to_node: dict[str, TreeNode] = {}

    # construct tree of instance topology
    for instance_info in dlcalc.utils.cluster.ec2.iter_instance_info(
        ec2_client=ec2_client,
        accepted_node_instance_types=accepted_node_instance_types,
        accepted_node_availability_zones=accepted_node_availability_zones,
        accepted_instance_ids=accepted_instance_ids,
    ):
        if instance_info.instance_id in node_id_to_node:
            raise AssertionError(
                f"instance {instance_info.instance_id} has been visited twice. "
                f"check for duplicate nodes from EC2 API or cycles in connectivity "
                f"(which are not supported)"
            )

        # order of network nodes: farthest from instance -> closest to instance
        chain = [*instance_info.network_nodes, instance_info.instance_id]

        # update nodes
        for node_id in chain:
            if node_id not in node_id_to_node:
                node_id_to_node[node_id] = TreeNode(node_id, parent=None, children=set())

        # update edges
        for parent_node_idx in range(len(chain) - 1):
            src_node_id = chain[parent_node_idx]
            dst_node_id = chain[parent_node_idx + 1]

            # set parent -> child relationship
            parent_node = node_id_to_node[src_node_id]
            child_node = node_id_to_node[dst_node_id]

            if child_node.parent is None:
                child_node.parent = parent_node
            elif child_node.parent != parent_node:
                raise AssertionError("mismatched parents")  # TODO. better error message
            parent_node.children.add(child_node)

    # some sanity checks
    root_nodes: set[TreeNode] = set()
    nonroot_nodes: set[TreeNode] = set()
    all_nodes: set[TreeNode] = set()
    for node in node_id_to_node.values():
        if node.parent is None:
            root_nodes.add(node)
        else:
            nonroot_nodes.add(node)
        all_nodes.add(node)

    if len(root_nodes) != 1:
        raise RuntimeError(
            f"expected to find 1 root node out of {len(all_nodes)} nodes but got {len(root_nodes)}"
        )
    for node in nonroot_nodes:
        assert node.parent in all_nodes

    return node_id_to_node


def dfs_tree_leaves_only(root: TreeNode) -> list[TreeNode]:
    if not root.children:
        return [root]

    list_of_list_of_leaves = []
    for child in root.children:
        list_of_list_of_leaves.append(dfs_tree_leaves_only(child))

    # now we have a list of list of leaves acquired from each subtree.
    # we want this list of lists of leaves to be ordered from largest list -> smallest list
    # this means we use the largest subtrees first - meaning that we will prioritize
    # groups of instances that can communicate with maximum "locality"
    list_of_list_of_leaves.sort(key=len, reverse=True)

    # stick everything together. this gives a prioritized order of instances
    # that should be part of the same ring.
    list_of_leaves = list(itertools.chain.from_iterable(list_of_list_of_leaves))

    return list_of_leaves


@functools.cache
def distance_between_nodes(node_a: TreeNode, node_b: TreeNode) -> int:
    if node_a == node_b:
        return 0

    return 2 + distance_between_nodes(node_a.parent, node_b.parent)


def traversal_distance_of_ring(ring_participants: list[TreeNode]) -> int:
    traversed_distance = 0
    for src_node_idx in range(len(ring_participants)):
        dst_node_idx = (src_node_idx + 1) % len(ring_participants)

        src_node = ring_participants[src_node_idx]
        dst_node = ring_participants[dst_node_idx]

        step_distance = distance_between_nodes(src_node, dst_node)

        traversed_distance += step_distance

    return traversed_distance
