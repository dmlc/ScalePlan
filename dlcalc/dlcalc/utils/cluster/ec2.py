from __future__ import annotations

import dataclasses
from collections.abc import Iterator

from botocore.client import BaseClient  # type: ignore[import-untyped]


@dataclasses.dataclass
class InstanceInfo:
    instance_id: str
    instance_type: str
    instance_az: str
    network_nodes: list[str]  # order: farthest from instance -> closest to instance


def iter_instance_info(  # type: ignore[no-any-unimported]
    ec2_client: BaseClient,
    accepted_node_instance_types: set[str],
    accepted_node_availability_zones: set[str],
    accepted_instance_ids: set[str] | None = None,
) -> Iterator[InstanceInfo]:
    # ref: https://docs.aws.amazon.com/cli/latest/reference/ec2/describe-instance-topology.html
    paginator = ec2_client.get_paginator("describe_instance_topology")

    # we have to filter in this roundabout way instead of giving a set of instance IDs
    # because API doesn't accept >100 instance IDs
    instance_filters = [
        {
            "Name": "instance-type",
            "Values": list(accepted_node_instance_types),
        },
        {
            "Name": "availability-zone",
            "Values": list(accepted_node_availability_zones),
        },
    ]

    for page in paginator.paginate(Filters=instance_filters, PaginationConfig={"PageSize": 100}):
        for instance_info in page["Instances"]:
            if (
                accepted_instance_ids is None
                or instance_info["InstanceId"] in accepted_instance_ids
            ):
                yield InstanceInfo(
                    instance_id=instance_info["InstanceId"],
                    instance_type=instance_info["InstanceType"],
                    instance_az=instance_info["AvailabilityZone"],
                    network_nodes=instance_info["NetworkNodes"],
                )
